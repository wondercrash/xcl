//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    /// a bind waits (or waited) for it: its build goes ahead of others for a big-compile slot
    wanted: std::sync::atomic::AtomicBool,
}

impl Pending {
    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Arc<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    complete(&p, build(&p, job));
                }
            });
        }
        Mutex::new(tx)
    })
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built ahead (`Preload`) or else at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`),
/// and for every compute pipeline (after those: `Preload`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        wanted: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        let _ = queue().lock().unwrap().send(p);
    } else if used::known(key) || compute {
        if let Some(q) = preload() {
            let mut g = q.list.lock().unwrap();
            if used::known(key) {
                g.0.push_back(p);
            } else {
                g.1.push_back(p);
            }
            drop(g);
            q.cv.notify_one();
        }
    }
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it, and the build of one a bind waits for goes first for a big-compile
/// slot: `big_compile`), and the title's own threads are not either.
/// After those, every compute pipeline is built ahead too, as it is made: its first bind waits
/// for it on the title's thread (a skipped dispatch would corrupt what it computes), and the
/// driver takes seconds for a big one – on an NVIDIA GPU a retail title's binds waited 3 – 23 s
/// each, 16 of them within 10 s on five of its threads, and its hang watchdog ended it. Compute
/// pipelines are few (2 734 of the title's ~58 000), so building them all costs little; graphics
/// pipelines stay lazy. Each build is kept on disk (`real::driver_cache`) for later runs.
struct Preload {
    /// (of shaders an earlier run used, the other compute pipelines): the first goes first
    list: Mutex<(std::collections::VecDeque<Arc<Pending>>, std::collections::VecDeque<Arc<Pending>>)>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.0.pop_front().or_else(|| g.1.pop_front()) {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        let real = build(&p, job);
                        complete(&p, real);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

thread_local! {
    /// the pipeline this thread builds now (`build`)
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// Runs the build of `p` on this thread, known to `big_compile` meanwhile.
fn build(p: &Arc<Pending>, job: Job) -> usize {
    let outer = BUILDING.with(|b| b.replace(Some(p.clone())));
    let real = job();
    BUILDING.with(|b| *b.borrow_mut() = outer);
    real
}

/// A bind waits for `p`: if its build waits for a big-compile slot, it goes ahead of the others.
fn want(p: &Pending) {
    if !p.wanted.load(Ordering::Acquire) && !p.wanted.swap(true, Ordering::AcqRel) {
        // under the lock: a waiter checks `wanted` under it before it sleeps
        let _g = BIG_COMPILES.state.lock().unwrap();
        BIG_COMPILES.cv.notify_all();
    }
}

/// A counting semaphore: driver compiles of big programs at once. A thread building a pipeline
/// a bind waits for (`wanted`) gets the next free slot ahead of the others waiting (the
/// pipelines built ahead of their first use: a preload thread takes a job, then waits here).
struct Slots {
    state: Mutex<SlotState>,
    cv: Condvar,
    most: usize,
}

struct SlotState {
    /// slots taken
    held: usize,
    /// the pipelines whose builds wait for one (the thread's `BUILDING`); a bind marks one
    /// `wanted` without waking its thread first, so the others look here
    waiting: Vec<Arc<Pending>>,
}

/// A slot, given back when dropped (and this thread's priority with it).
pub struct SlotGuard {
    slots: &'static Slots,
    /// priority the thread had before it was raised for a build a bind waits for
    priority: Option<i32>,
}

impl Slots {
    const fn new(most: usize) -> Self {
        Slots { state: Mutex::new(SlotState { held: 0, waiting: Vec::new() }), cv: Condvar::new(), most }
    }

    /// Waits for a slot for the build of `mine` (None: no pipeline's). A build a bind waits for
    /// (`wanted`, read again at each wake-up) takes the next free slot; the others wait while
    /// one of those waits – even one whose thread has not woken up since its bind.
    fn acquire(&'static self, mine: Option<Arc<Pending>>) -> SlotGuard {
        let wanted = |p: &Pending| p.wanted.load(Ordering::Acquire);
        let mut g = self.state.lock().unwrap();
        if let Some(p) = &mine {
            g.waiting.push(p.clone());
        }
        let ahead = loop {
            if g.held < self.most {
                let ahead = mine.as_deref().is_some_and(wanted);
                if ahead || !g.waiting.iter().any(|q| wanted(q)) {
                    break ahead;
                }
            }
            g = self.cv.wait(g).unwrap();
        };
        if let Some(p) = &mine {
            let i = g.waiting.iter().position(|q| Arc::ptr_eq(q, p)).unwrap();
            g.waiting.swap_remove(i);
        }
        g.held += 1;
        let passed = g.waiting.iter().filter(|q| !wanted(q)).count();
        drop(g);
        if !ahead {
            return SlotGuard { slots: self, priority: None };
        }
        // (the others may go now if a slot is left)
        self.cv.notify_all();
        if passed > 0 {
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n < 16 || n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: a big pipeline a bind waits for is compiled ahead of {passed} others ({} so far)", n + 1);
            }
        }
        // a preload thread runs at the lowest priority: not while the title waits for its build
        let priority = unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY_NORMAL};
            // (a failed call returns THREAD_PRIORITY_ERROR_RETURN, i32::MAX: not below normal)
            let p = GetThreadPriority(GetCurrentThread());
            (p < THREAD_PRIORITY_NORMAL && SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) != 0).then_some(p)
        };
        SlotGuard { slots: self, priority }
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if let Some(p) = self.priority {
            unsafe {
                use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
                SetThreadPriority(GetCurrentThread(), p);
            }
        }
        self.slots.state.lock().unwrap().held -= 1;
        self.slots.cv.notify_all();
    }
}

/// Driver compiles of big programs in the background: one at a time (each ~1.5 GB of commit in
/// the driver; `real::wait_for_commit` under the slot).
static BIG_COMPILES: Slots = Slots::new(1);

/// A slot for a big driver compile on a worker; the build of a pipeline a bind waits for gets
/// the next free one, ahead of the pipelines built ahead of their first use.
pub fn big_compile() -> SlotGuard {
    BIG_COMPILES.acquire(BUILDING.with(|b| b.borrow().clone()))
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// A compute pipeline is built without a bind even when no earlier run used it: its first
    /// bind would otherwise build it on the title's thread. A graphics one made with it is not.
    #[test]
    fn a_compute_pipeline_is_built_ahead_of_its_first_bind() {
        let (compute, graphics) = (0x7E57_7003usize, 0x7E57_7004usize);
        submit_lazy(compute, true, Box::new(|| 0x4323), 0x5EED_0003);
        submit_lazy(graphics, false, Box::new(|| 0x4324), 0x5EED_0004);
        let start = std::time::Instant::now();
        while is_pending(compute) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(compute), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(compute), 0x4323);
        assert!(is_pending(graphics), "graphics pipelines stay lazy");
        assert_eq!(resolve_wait(graphics), 0x4324);
        released(compute);
        released(graphics);
    }

    /// A bind of a pipeline whose build a preload thread took and that waits for the one
    /// big-compile slot with other pipelines built ahead gets the next free slot, ahead of them.
    #[test]
    fn a_bound_pipeline_gets_the_next_big_compile_slot_ahead_of_the_ones_built_ahead() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let arrived = Arc::new(Mutex::new(Vec::new()));
        let job = |id: usize| -> Job {
            let (order, arrived) = (order.clone(), arrived.clone());
            Box::new(move || {
                arrived.lock().unwrap().push(id);
                let _slot = big_compile();
                order.lock().unwrap().push(id);
                std::thread::sleep(std::time::Duration::from_millis(5));
                id ^ 0xF000
            })
        };
        // held here until the bind waits: the preload threads queue for it
        let held = big_compile();
        let bound = 0x7E57_8000usize;
        submit_lazy(bound, true, job(bound), 0);
        let others: Vec<usize> = (1..=8).map(|k| bound + k).collect();
        for &o in &others {
            submit_lazy(o, true, job(o), 0);
        }
        let p = entry(bound).unwrap();
        let start = std::time::Instant::now();
        // every build that arrived at the slot (the bound one too) waits for it
        loop {
            let (bound_arrived, n) = {
                let a = arrived.lock().unwrap();
                (a.contains(&bound), a.len())
            };
            if bound_arrived && BIG_COMPILES.state.lock().unwrap().waiting.len() == n {
                break;
            }
            assert!(start.elapsed() < std::time::Duration::from_secs(10), "a preload thread takes the bound pipeline");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(p.job.lock().unwrap().is_none(), "its job is taken: the bind cannot build it itself");
        let bind = std::thread::spawn(move || resolve_wait_for(bound, None));
        while !p.wanted.load(Ordering::Acquire) {
            assert!(start.elapsed() < std::time::Duration::from_secs(10));
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        drop(held);
        assert_eq!(bind.join().unwrap(), bound ^ 0xF000);
        let first = order.lock().unwrap().first().copied();
        assert_eq!(first, Some(bound), "the bound pipeline went first");
        for &o in &others {
            while is_pending(o) && start.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            released(o);
        }
        released(bound);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            if !p.sent.swap(true, Ordering::AcqRel) {
                let _ = queue().lock().unwrap().send(p);
            }
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    want(&p);
    if let Some(budget) = budget {
        return resolve_within(&p, pso, budget);
    }
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        complete(&p, build(&p, job));
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset: until it
/// is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(std::time::Duration::from_millis))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        let _ = queue().lock().unwrap().send(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return pso;
    }
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    if *done {
        return built(p);
    }
    drop(done);
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    pso
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
