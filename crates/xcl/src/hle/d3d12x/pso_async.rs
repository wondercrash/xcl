//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

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
    /// the stand-in the title has (what the logs name it by)
    standin: usize,
    /// where its build is, for the logs of a bind that waits for it
    stage: Mutex<Stage>,
    /// its DXIL size once translated (`dxil_known`; 0: not yet)
    dxil: AtomicUsize,
    /// a hash of its console shader (`shader::fnv64`) once translated (0: not yet)
    hash: AtomicU64,
    /// its DXIL goes to the driver's compiler (no blob of an earlier build), known with `dxil`
    compiles: AtomicBool,
    /// its place in the workers' queue (`send`; 0: not sent)
    seq: AtomicUsize,
    /// it went to a list of `Preload`
    listed: AtomicBool,
    /// a bind's long wait for it was logged (once a pipeline)
    wait_logged: AtomicBool,
    /// a bind went on without it for `XCL_PSO_SKIP_HUGE` (logged once a pipeline)
    skip_logged: AtomicBool,
    demand: Mutex<Demand>,
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// Whether one of the title's threads waits for a worker's build (`real::big_compile_turn`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Want {
    /// no: built ahead of its first use, or binds go on without it for its size
    /// (`XCL_PSO_SKIP_HUGE`: it takes its turn like those built ahead)
    No,
    /// a bind waited for it longer than `XCL_PSO_WAIT_MS` (`late`): binds go on without it now
    Late,
    /// a bind waits for it: the build must not stand back for the title's sake
    Waits,
}

/// `Want` of what this thread builds (`No`: nothing).
pub(super) fn build_want() -> Want {
    BUILDING.with(|b| b.borrow().as_ref().map_or(Want::No, |p| p.want_now(skip_huge())))
}

/// This thread is a worker and one of the title's threads waits, or waited, for what it builds.
#[cfg(test)]
fn build_is_wanted() -> bool {
    build_want() != Want::No
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build, once a bind has waited for it; a thread that is not
/// XCL's is the title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted);
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A build of `p` on this thread – a worker's (`preload`: by one of the lowest-priority
/// workers), or a bind's – known to `build_want`, `dxil_known` and the logs meanwhile: its
/// pipeline (0 = failed).
fn build(p: &Arc<Pending>, job: Job, preload: bool) -> usize {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    p.set_stage(Stage::Building(Instant::now(), Who::me()));
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    let low = preload && preload_runs_low();
    if low {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    let real = job();
    complete(p, real);
    if low {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
    real
}

/// The preload workers run at the lowest priority, raised to normal while they build a pipeline a
/// bind waits for (`build`, `Pending::want`). Not under Wine: it applies a thread priority as a
/// Linux nice value, and raising one again needs `RLIMIT_NICE` / `CAP_SYS_NICE`, which a desktop
/// user does not have – a worker lowered once would build the pipelines binds wait for at the
/// lowest priority for the rest of the run.
fn preload_runs_low() -> bool {
    !crate::util::under_wine()
}

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let mut d = self.demand.lock().unwrap();
        if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
    }

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
    summary();
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
                // A released stand-in no longer needs a queued build (as in preload).
                if p.orphan.load(Ordering::Acquire) {
                    continue;
                }
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    build(&p, job, false);
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
        standin,
        stage: Mutex::new(Stage::Queued),
        dxil: AtomicUsize::new(0),
        hash: AtomicU64::new(0),
        compiles: AtomicBool::new(false),
        seq: AtomicUsize::new(0),
        listed: AtomicBool::new(false),
        wait_logged: AtomicBool::new(false),
        skip_logged: AtomicBool::new(false),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        send_to_workers(p);
    } else if used::known(key) || compute {
        if let Some(q) = preload() {
            p.listed.store(true, Ordering::Relaxed);
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
/// `job` first builds it, and the build of one a bind waits for does not wait for its turn at a
/// big-compile slot: `real::big_compile_turn`), and the title's own threads are not either.
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
                if preload_runs_low() {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
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
                        build(&p, job, true);
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
        assert_eq!(resolve_wait_for(standin, budget, None), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget, None), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        assert_eq!(entry(standin).unwrap().want_now(None), Want::Late, "its build goes without a slot only a few at a time (`real::slotless_goes`)");
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget, None), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// With a size limit (`XCL_PSO_SKIP_HUGE`) a bind waits for a pipeline until its DXIL is known
    /// to go to the driver's compiler with that many bytes or more, then goes on with the stand-in
    /// – with or without a budget, and later binds do not wait – while the build goes on; once
    /// built it is the real one. A smaller one, and a huge one the driver has a blob of, are
    /// waited for until built.
    #[test]
    fn a_bind_does_not_wait_for_a_huge_pipeline() {
        let limit = Some(8 << 20);
        let (huge, small, blob) = (0x7E57_9001usize, 0x7E57_9002usize, 0x7E57_9003usize);
        let (go, driver) = mpsc::channel::<()>();
        submit_lazy(huge, true, Box::new(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            dxil_known(29_750_144, b"huge", true);
            // the driver's compile, until the test lets it end
            let _ = driver.recv_timeout(std::time::Duration::from_secs(10));
            0x9001
        }), 0);
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(huge, None, limit), huge, "the stand-in once its size is known");
        assert!(t.elapsed() < std::time::Duration::from_secs(5), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(huge, None, limit), huge);
        assert_eq!(resolve_wait_for(huge, Some(std::time::Duration::from_secs(5)), limit), huge);
        assert!(t.elapsed() < std::time::Duration::from_millis(500), "not waited for again ({:?})", t.elapsed());
        let p = entry(huge).unwrap();
        assert!(is_pending(huge) && !p.late.load(Ordering::Acquire), "still being built; no budget ran out");
        assert!(p.skip_logged.load(Ordering::Acquire));
        let wanted = p.demand.lock().unwrap().wanted;
        assert!(wanted && p.want_now(limit) == Want::No, "its driver compile takes its turn: no bind waits for it");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(huge) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(huge, None, limit), 0x9001, "built in the background");
        let slow = |dxil: usize, compiles: bool, real: usize| -> Job {
            Box::new(move || {
                dxil_known(dxil, b"console", compiles);
                std::thread::sleep(std::time::Duration::from_millis(100));
                real
            })
        };
        submit_lazy(small, true, slow(1 << 20, true, 0x9002), 0);
        submit_lazy(blob, true, slow(29_750_144, false, 0x9003), 0);
        assert_eq!(resolve_wait_for(small, None, limit), 0x9002, "a smaller one is waited for");
        assert_eq!(resolve_wait_for(blob, None, limit), 0x9003, "one with a driver blob is waited for");
        released(huge);
        released(small);
        released(blob);
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None, None), 0x5353, "the bind's wait reached the build");
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

    /// A bind of a pipeline whose build a preload thread took and that waits for its turn at the
    /// big-compile slot a pipeline built ahead holds (a retail game's biggest programs held it for
    /// many minutes) is built at once, without waiting for that one or the other ones built ahead.
    #[test]
    fn a_bound_pipeline_does_not_wait_for_the_big_compiles_of_the_ones_built_ahead() {
        use super::super::real::{big_compile_turn, BIG_COMPILES};
        let order = Arc::new(Mutex::new(Vec::new()));
        let arrived = Arc::new(Mutex::new(Vec::new()));
        let job = |id: usize| -> Job {
            let (order, arrived) = (order.clone(), arrived.clone());
            Box::new(move || {
                arrived.lock().unwrap().push(id);
                let _slot = big_compile_turn(&BIG_COMPILES, || true, build_is_wanted);
                order.lock().unwrap().push(id);
                std::thread::sleep(std::time::Duration::from_millis(5));
                id ^ 0xF000
            })
        };
        // held here for the whole test, as by a build ahead the driver never finishes: the
        // preload threads queue for it
        let held = big_compile_turn(&BIG_COMPILES, || true, || false).expect("a free slot");
        let bound = 0x7E57_8000usize;
        submit_lazy(bound, true, job(bound), 0);
        let others: Vec<usize> = (1..=8).map(|k| bound + k).collect();
        for &o in &others {
            submit_lazy(o, true, job(o), 0);
        }
        let p = entry(bound).unwrap();
        let start = std::time::Instant::now();
        // the bound one's build waits for its turn
        while !arrived.lock().unwrap().contains(&bound) {
            assert!(start.elapsed() < std::time::Duration::from_secs(10), "a preload thread takes the bound pipeline");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(p.job.lock().unwrap().is_none(), "its job is taken: the bind cannot build it itself");
        let bind = std::thread::spawn(move || resolve_wait_for(bound, None, None));
        while !bind.is_finished() {
            // (the slot is given back on the failure's unwinding, and the bind ends then)
            assert!(start.elapsed() < std::time::Duration::from_secs(10), "the bind waits for the slot a pipeline built ahead holds");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(bind.join().unwrap(), bound ^ 0xF000);
        let first = order.lock().unwrap().first().copied();
        assert_eq!(first, Some(bound), "the bound pipeline went first");
        drop(held);
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
                send_to_workers(p);
            }
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget(), skip_huge())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built), and not once its
/// DXIL is known to go to the driver's compiler with `huge` bytes or more (None: no limit).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>, huge: Option<usize>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    p.want();
    // only a call that waits is counted: not one for a pipeline built already, late
    // (`XCL_PSO_WAIT_MS`) or known to be too big to wait for (`XCL_PSO_SKIP_HUGE`)
    let waits = !*p.done.lock().unwrap() && !p.late.load(Ordering::Acquire) && !p.too_big(huge);
    let started = Instant::now();
    if let Some(budget) = budget {
        let r = resolve_within(&p, pso, budget, huge);
        if waits {
            note_blocked(p.compute, false, started.elapsed());
        }
        return r;
    }
    // (with a size limit, built on a thread of its own: this one may go on without it)
    let aside = huge.is_some() && p.job.lock().unwrap().is_some() && build_aside(&p);
    let job = if aside { None } else { p.job.lock().unwrap().take() };
    let built_here = job.is_some();
    if let Some(job) = job {
        // built on this thread: as long a wait
        let start = Instant::now();
        WAITING.fetch_add(1, Ordering::Relaxed);
        let real = build(&p, job, false);
        WAITING.fetch_sub(1, Ordering::Relaxed);
        if start.elapsed() >= LONG_WAIT {
            crate::xlog_info!(
                "D3D12.X: a bind waited {:.1} s in all for {}: {} on the binding thread ({})",
                start.elapsed().as_secs_f64(),
                p.describe(),
                if real == 0 { "its build failed" } else { "built" },
                Who::me()
            );
        }
    }
    wait_built(&p, None, huge);
    if waits {
        note_blocked(p.compute, built_here, started.elapsed());
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// The build of `p` a bind would run on its own thread, on a new thread instead (named so that it
/// takes its turn like a worker's, `real::big_compile_turn`: at once while its bind waits, at its
/// lane once binds go on without it): whether one was started.
fn build_aside(p: &Arc<Pending>) -> bool {
    let p = p.clone();
    std::thread::Builder::new()
        .name("xcl-bind-build".into())
        .stack_size(64 << 20)
        .spawn(move || {
            // (None: a worker took it meanwhile)
            let job = p.job.lock().unwrap().take();
            if let Some(job) = job {
                build(&p, job, false);
            }
        })
        .is_ok()
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// A bring-up switch (`XCL_PSO_SKIP_HUGE=<bytes>`, e.g. `8M`; `shaders::parse_bytes`; unset: off):
/// a bind does not wait for a pipeline not built yet whose DXIL (known once translated, before
/// the driver's compile) is at least this big and goes to the driver's compiler (no blob of an
/// earlier build) – it goes on with the stand-in (its dispatches or draws are skipped) while the
/// build goes on in the background, and a later bind uses it once built. Smaller ones are waited
/// for as before; with `XCL_PSO_WAIT_MS` both apply. On an NVIDIA GPU the driver never finished
/// a retail title's 29.75 MB compute program, and its bind at a cutscene waited for good.
fn skip_huge() -> Option<usize> {
    static H: OnceLock<Option<usize>> = OnceLock::new();
    *H.get_or_init(|| std::env::var("XCL_PSO_SKIP_HUGE").ok().and_then(|v| crate::shaders::parse_bytes(&v)).filter(|&n| n > 0))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration, huge: Option<usize>) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        send_to_workers(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return pso;
    }
    if wait_built(p, Some(budget), huge) {
        return built(p);
    }
    if p.too_big(huge) {
        // (logged by `wait_built`; not late: the budget did not run out)
        return pso;
    }
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    pso
}

/// What `resolve_wait` cost the threads that called it – a title's, as a rule: for graphics
/// [0] and compute [1] pipelines, the waits, their total and longest µs, and those that built
/// the pipeline themselves. A retail title's render thread spent most of its frames in the
/// driver's compiler this way (on an NVIDIA GPU; its hang watchdog then ended it).
static BLOCKED: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];
const WAITS: usize = 0;
const TOTAL_US: usize = 1;
const LONGEST_US: usize = 2;
const BUILT_HERE: usize = 3;

/// (waits, total µs, longest µs, built by the caller) of `resolve_wait`, graphics then compute
pub fn blocked() -> [[u64; 4]; 2] {
    BLOCKED.each_ref().map(|k| k.each_ref().map(|c| c.load(Ordering::Relaxed)))
}

/// A `resolve_wait` that found its pipeline not built yet: counted, and logged at every power of
/// two of them (a long wait itself is logged by `wait_built` and on the binding thread).
fn note_blocked(compute: bool, built_here: bool, took: std::time::Duration) {
    let us = took.as_micros() as u64;
    let k = &BLOCKED[compute as usize];
    let n = k[WAITS].fetch_add(1, Ordering::Relaxed) + 1;
    k[TOTAL_US].fetch_add(us, Ordering::Relaxed);
    k[LONGEST_US].fetch_max(us, Ordering::Relaxed);
    if built_here {
        k[BUILT_HERE].fetch_add(1, Ordering::Relaxed);
    }
    if n.is_power_of_two() {
        let b = blocked();
        crate::xlog_info!(
            "D3D12.X: pipelines waited for at a bind so far: {} graphics ({} ms, longest {} ms, {} built in the call), {} compute ({} ms, longest {} ms, {} built in the call)",
            b[0][WAITS], b[0][TOTAL_US] / 1000, b[0][LONGEST_US] / 1000, b[0][BUILT_HERE],
            b[1][WAITS], b[1][TOTAL_US] / 1000, b[1][LONGEST_US] / 1000, b[1][BUILT_HERE],
        );
    }
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

/// To the workers (`queue`), numbered in their order for the logs (`Pending::seq`).
fn send_to_workers(p: Arc<Pending>) {
    static SENT: AtomicUsize = AtomicUsize::new(0);
    let q = queue().lock().unwrap();
    p.seq.store(SENT.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    let _ = q.send(p);
}

/// A bind that has waited this long for a pipeline logs where its build is (`wait_built`).
const LONG_WAIT: Duration = Duration::from_secs(2);
/// `summary` at most this often
const SUMMARY_EVERY: Duration = Duration::from_secs(5);
/// binds waiting for a pipeline now (`wait_built`, or building it themselves)
static WAITING: AtomicUsize = AtomicUsize::new(0);

/// A thread, as the logs and a debugger's thread list name it.
#[derive(Clone)]
struct Who {
    thread: std::thread::Thread,
    tid: u32,
}

impl Who {
    fn me() -> Who {
        Who { thread: std::thread::current(), tid: unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() } }
    }
}

impl std::fmt::Display for Who {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.thread.name() {
            Some(name) => write!(f, "{name} (thread {:x})", self.tid),
            None => write!(f, "thread {:x}", self.tid),
        }
    }
}

/// Where a pipeline's build is (`Pending::stage`; each since when, on which thread).
#[derive(Clone)]
enum Stage {
    /// no thread has taken it yet
    Queued,
    /// a thread translates it, compiles its HLSL (DXC) or creates it
    Building(Instant, Who),
    /// its build waits for its turn at this lane of big driver compiles (`real::big_compile_turn`)
    Slot(Instant, Who, &'static str),
    /// its build holds a slot and waits for free commit (`real::big_compile_commit_free`)
    Commit(Instant, Who),
    /// in the driver's compiler (`driver_start`), holding a slot of this lane (None: without one)
    Driver(Instant, Who, Option<&'static str>),
}

fn slot_text(slot: Option<&'static str>) -> String {
    match slot {
        Some(lane) => format!("holding the {lane} of the builds no bind waited for"),
        None => "without a compile slot".into(),
    }
}

impl Pending {
    fn set_stage(&self, stage: Stage) {
        *self.stage.lock().unwrap() = stage;
    }

    /// Its DXIL goes to the driver's compiler and is `huge` bytes or more (None: no limit;
    /// `skip_huge`).
    fn too_big(&self, huge: Option<usize>) -> bool {
        huge.is_some_and(|h| self.compiles.load(Ordering::Acquire) && self.dxil.load(Ordering::Relaxed) >= h)
    }

    /// Whether a bind waits for it now (`Want`; `huge`: `skip_huge`).
    fn want_now(&self, huge: Option<usize>) -> Want {
        if !self.demand.lock().unwrap().wanted || self.too_big(huge) {
            Want::No
        } else if self.late.load(Ordering::Acquire) {
            Want::Late
        } else {
            Want::Waits
        }
    }

    /// "compute pipeline 0x… (shader set …, console shader …, N B DXIL)": what is known of it.
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.key != 0 {
            parts.push(format!("shader set {:016x}", self.key));
        }
        match self.hash.load(Ordering::Relaxed) {
            0 => {}
            h => parts.push(format!("console shader {h:016x}")),
        }
        parts.push(match self.dxil.load(Ordering::Relaxed) {
            0 => "DXIL size not known yet".into(),
            n => format!("{n} B DXIL"),
        });
        format!("{} pipeline {:#x} ({})", if self.compute { "compute" } else { "graphics" }, self.standin, parts.join(", "))
    }

    /// Where its build is now. Takes PENDING when it is queued: not called under it.
    fn whereabouts(self: &Arc<Self>) -> String {
        let stage = self.stage.lock().unwrap().clone();
        let secs = |t: Instant| t.elapsed().as_secs_f64();
        match stage {
            Stage::Queued => self.queue_place(),
            Stage::Building(t, who) if self.compute => format!("translated / compiled by DXC on {who} for {:.1} s, not in the driver's compiler yet", secs(t)),
            Stage::Building(t, who) => format!("being built on {who} for {:.1} s (translation, DXC or the driver)", secs(t)),
            Stage::Slot(t, who, lane) => format!("waiting {:.1} s on {who} for its turn at the {lane} (one at a time; at once when a bind waits for it)", secs(t)),
            Stage::Commit(t, who) => format!("holding a compile slot on {who}, waiting {:.1} s for 3 GB of free commit before the driver's compiler", secs(t)),
            Stage::Driver(t, who, slot) => format!("in the driver's compiler for {:.1} s on {who}, {}", secs(t), slot_text(slot)),
        }
    }

    /// "not started: N ahead of it in the workers' queue; M ahead of it among those built ahead".
    fn queue_place(self: &Arc<Self>) -> String {
        let mut parts = Vec::new();
        let seq = self.seq.load(Ordering::Relaxed);
        if seq != 0 {
            let ahead = PENDING.lock().unwrap().as_ref().map_or(0, |m| {
                m.values()
                    .filter(|q| {
                        let s = q.seq.load(Ordering::Relaxed);
                        s != 0 && s < seq && q.job.lock().unwrap().is_some()
                    })
                    .count()
            });
            parts.push(format!("{ahead} ahead of it in the workers' queue ({} workers)", crate::gpu::recomp::dxc::compile_threads()));
        }
        if let (true, Some(q)) = (self.listed.load(Ordering::Relaxed), preload()) {
            let g = q.list.lock().unwrap();
            let list = g.0.iter().chain(g.1.iter());
            if let Some(i) = list.clone().position(|q| Arc::ptr_eq(q, self)) {
                let ahead = list.take(i).filter(|q| q.job.lock().unwrap().is_some()).count();
                parts.push(format!("{ahead} ahead of it among those built ahead of their first use"));
            }
        }
        if parts.is_empty() {
            return "not started".into();
        }
        format!("not started: {}", parts.join("; "))
    }
}

/// Waits until `p` is built or `budget` has passed (None: no limit), or until it is known to be
/// `huge` (`Pending::too_big`; logged once a pipeline): whether it was built.
/// A wait of `LONG_WAIT` logs where the build is (the first for a pipeline) and, then, its end.
fn wait_built(p: &Arc<Pending>, budget: Option<Duration>, huge: Option<usize>) -> bool {
    let mut done = p.done.lock().unwrap();
    if *done {
        return true;
    }
    let start = Instant::now();
    WAITING.fetch_add(1, Ordering::Relaxed);
    let (mut long, mut logged) = (false, false);
    loop {
        let waited = start.elapsed();
        if budget.is_some_and(|b| waited >= b) || p.too_big(huge) {
            break;
        }
        let tick = if long { Duration::from_secs(1) } else { LONG_WAIT.saturating_sub(waited) }.max(Duration::from_millis(1));
        let step = budget.map_or(tick, |b| tick.min(b - waited));
        done = p.cv.wait_timeout_while(done, step, |d| !*d && !p.too_big(huge)).unwrap().0;
        if *done {
            break;
        }
        if start.elapsed() >= LONG_WAIT {
            // (not under `done`: these take PENDING, which is taken before `done`)
            drop(done);
            if !long {
                long = true;
                logged = !p.wait_logged.swap(true, Ordering::AcqRel);
                if logged {
                    crate::xlog_warn!("D3D12.X: a bind has waited {:.1} s for {}: {}", start.elapsed().as_secs_f64(), p.describe(), p.whereabouts());
                }
            }
            summary();
            done = p.done.lock().unwrap();
        }
    }
    let built = *done;
    drop(done);
    WAITING.fetch_sub(1, Ordering::Relaxed);
    let skipped = !built && p.too_big(huge);
    if skipped && !p.skip_logged.swap(true, Ordering::AcqRel) {
        crate::xlog_warn!(
            "D3D12.X: XCL_PSO_SKIP_HUGE: {} pipeline {:#x} (console shader {:016x}, {} B DXIL) is not waited for; its {} are skipped until it is built",
            if p.compute { "compute" } else { "graphics" },
            p.standin,
            p.hash.load(Ordering::Relaxed),
            p.dxil.load(Ordering::Relaxed),
            if p.compute { "dispatches" } else { "draws" }
        );
    }
    if logged {
        let end = match (built, p.real.load(Ordering::Acquire)) {
            (false, _) if skipped => "its DXIL is at least XCL_PSO_SKIP_HUGE: its binds go on without it",
            (false, _) => "not built within XCL_PSO_WAIT_MS: its binds go on without it",
            (true, 0) => "its build failed",
            (true, _) => "built",
        };
        crate::xlog_info!("D3D12.X: a bind waited {:.1} s in all for {}: {end}", start.elapsed().as_secs_f64(), p.describe());
    }
    built
}

/// The pipeline this thread builds is translated: `dxil` bytes of DXIL (all its shaders) for
/// console shader `console` (a graphics one's VS); `compiles`: the driver's compiler takes it
/// next (no blob of an earlier build). A bind waiting for it may go on without it now
/// (`skip_huge`).
pub fn dxil_known(dxil: usize, console: &[u8], compiles: bool) {
    let Some(p) = BUILDING.with(|b| b.borrow().clone()) else { return };
    p.dxil.store(dxil, Ordering::Relaxed);
    p.hash.store(crate::gpu::shader::fnv64(console), Ordering::Relaxed);
    p.compiles.store(compiles, Ordering::Release);
    // under `done`: a bind checks `too_big` under it before it sleeps
    let _g = p.done.lock().unwrap();
    p.cv.notify_all();
}

/// The build on this thread waits for its turn at `lane` of big driver compiles (for the logs).
pub(super) fn turn_wait(lane: &'static str) {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        p.set_stage(Stage::Slot(Instant::now(), Who::me(), lane));
    }
}

/// The build on this thread holds its slot and waits for free commit (for the logs; asked at
/// each look).
pub(super) fn commit_wait() {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        let mut stage = p.stage.lock().unwrap();
        if !matches!(*stage, Stage::Commit(..)) {
            *stage = Stage::Commit(Instant::now(), Who::me());
        }
    }
}

/// A driver compile of a compute program of `dxil` bytes (console shader `console`) starts on
/// this thread now, with `slot` (None: without one); a `big` one is logged. (A pipeline's build
/// told `dxil_known` its DXIL first.)
pub fn driver_start(dxil: usize, console: &[u8], slot: Option<&'static str>, big: bool) {
    let mine = BUILDING.with(|b| b.borrow().clone());
    if mine.is_none() && !big {
        return;
    }
    let who = Who::me();
    if let Some(p) = &mine {
        p.set_stage(Stage::Driver(Instant::now(), who.clone(), slot));
    }
    if !big {
        return;
    }
    let (what, waits) = match &mine {
        Some(p) => (p.describe(), if p.demand.lock().unwrap().wanted { "yes" } else { "no" }),
        None => (format!("a compute program in the title's call (console shader {:016x}, {dxil} B DXIL)", crate::gpu::shader::fnv64(console)), "the title's call does"),
    };
    crate::xlog_info!("D3D12.X: driver compile of {what} starts on {who}, {}; a bind waits for it: {waits}", slot_text(slot));
}

/// At most every `SUMMARY_EVERY`, while pipelines are being built, are queued or binds wait:
/// how many are where, and the one longest in the driver's compiler.
fn summary() {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    {
        let Ok(mut last) = LAST.try_lock() else { return };
        if last.is_some_and(|t| t.elapsed() < SUMMARY_EVERY) {
            return;
        }
        *last = Some(Instant::now());
    }
    let (mut driver, mut slot, mut building, mut workers, mut ahead) = (0, 0, 0, 0, 0);
    let mut longest: Option<(Instant, Arc<Pending>)> = None;
    if let Some(m) = PENDING.lock().unwrap().as_ref() {
        for p in m.values() {
            if *p.done.lock().unwrap() {
                continue;
            }
            let stage = p.stage.lock().unwrap().clone();
            match stage {
                Stage::Driver(t, ..) => {
                    driver += 1;
                    if longest.as_ref().map_or(true, |(l, _)| t < *l) {
                        longest = Some((t, p.clone()));
                    }
                }
                Stage::Slot(..) | Stage::Commit(..) => slot += 1,
                Stage::Building(..) => building += 1,
                // (taken, its thread not started yet; or lazy and never wanted)
                Stage::Queued if p.job.lock().unwrap().is_none() => {}
                Stage::Queued if p.seq.load(Ordering::Relaxed) != 0 => workers += 1,
                Stage::Queued if p.listed.load(Ordering::Relaxed) => ahead += 1,
                Stage::Queued => {}
            }
        }
    }
    let waiting = WAITING.load(Ordering::Relaxed);
    if driver + slot + building + workers + ahead + waiting == 0 {
        return;
    }
    let longest = longest.map_or(String::new(), |(_, p)| format!(" (longest: {}, {})", p.describe(), p.whereabouts()));
    crate::xlog_info!(
        "D3D12.X: pipelines being built: {driver} in the driver's compiler{longest}, {slot} waiting for a compile slot or free commit, {building} being translated / compiled; queued: {workers} for the workers, {ahead} to build ahead of their first use; {waiting} binds waiting; {} built, {} failed so far",
        BUILT.load(Ordering::Relaxed),
        FAILED.load(Ordering::Relaxed)
    );
}

#[cfg(test)]
mod queued_release_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for OpenGate {
        fn drop(&mut self) {
            *self.0.0.lock().unwrap() = true;
            self.0.1.notify_all();
        }
    }

    struct Dropped(mpsc::Sender<()>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    #[test]
    fn released_queued_pipelines_are_not_built() {
        let workers = crate::gpu::recomp::dxc::compile_threads();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open = OpenGate(gate.clone());
        let (started, waiting) = mpsc::channel();
        for k in 0..workers {
            let gate = gate.clone();
            let started = started.clone();
            submit(0xcafe_1000 + k, false, Box::new(move || {
                started.send(()).unwrap();
                let mut ready = gate.0.lock().unwrap();
                while !*ready {
                    ready = gate.1.wait(ready).unwrap();
                }
                0
            }));
        }
        for _ in 0..workers {
            waiting.recv_timeout(Duration::from_secs(10)).expect("worker reached gate");
        }
        let called = Arc::new(AtomicBool::new(false));
        let built = called.clone();
        let (dropped, finished) = mpsc::channel();
        let dropped = Dropped(dropped);
        submit(0xcafe_2000, false, Box::new(move || {
            built.store(true, Ordering::Release);
            drop(dropped);
            0
        }));
        assert_eq!(released(0xcafe_2000), 0);
        drop(open);
        finished.recv_timeout(Duration::from_secs(10)).expect("queued closure dropped");
        for k in 0..workers {
            assert_eq!(released(0xcafe_1000 + k), 0);
        }
        assert!(!called.load(Ordering::Acquire), "released job reached the compiler");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bind that waits for a lazy pipeline nobody has started builds it itself, and the time
    /// it was held up is counted under its kind (other tests wait too: increases are checked).
    #[test]
    fn a_wait_that_builds_the_pipeline_is_counted_with_its_time() {
        let standin = 0x7E57_C001usize;
        let before = blocked();
        submit_lazy(standin, false, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            0x4401
        }), 0);
        assert_eq!(resolve_wait(standin), 0x4401);
        let after = blocked();
        let (b, a) = (before[0], after[0]);
        assert!(a[WAITS] > b[WAITS], "the graphics wait is counted");
        assert!(a[BUILT_HERE] > b[BUILT_HERE], "built by the caller: nobody had started it");
        assert!(a[TOTAL_US] - b[TOTAL_US] >= 50_000, "its 50 ms are in the total");
        assert!(a[LONGEST_US] >= 50_000);
        released(standin);
    }

    /// A bind with a wait budget (`XCL_PSO_WAIT_MS`) is counted with the time it waited, not
    /// as built in the call: the build goes to the workers (other tests wait too: increases
    /// are checked).
    #[test]
    fn a_bounded_wait_is_counted_with_the_time_it_waited() {
        let standin = 0x7E57_C002usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(400));
            0x4402
        }));
        let before = blocked();
        let budget = Some(std::time::Duration::from_millis(30));
        assert_eq!(resolve_wait_for(standin, budget, None), standin, "not built within the budget: the stand-in");
        let after = blocked();
        let (b, a) = (before[1], after[1]);
        assert!(a[WAITS] > b[WAITS], "the compute wait is counted");
        assert!(a[TOTAL_US] - b[TOTAL_US] >= 30_000, "with the budget it waited");
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let _ = released(standin);
    }
}
