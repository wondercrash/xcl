//! `XCL_GPU_PROFILE=1`: the GPU time each console program takes.
//!
//! Each draw, dispatch and ExecuteIndirect of console code gets a timestamp query on its PC
//! list right before it and one right after it (and after its breadcrumb), and the pair is
//! resolved into a readback buffer right behind them. The resolve waits on the GPU for the op to
//! finish, so with the profile on a queue runs its ops one after the other. Nothing on the CPU
//! waits for them: a reporter thread looks at the pairs
//! every 100 ms, adds the ones the GPU has written to their program, and every 2 s logs the 15
//! programs that took the most GPU time in those 2 s (every 20 s: since the start), with their
//! calls, longest call, group counts and, for compute, thread group size, wave size and code size.
//!
//! A pair measures the GPU clock between the two stamps (ticks of a title queue's
//! `GetTimestampFrequency`): the time the op held its queue. Queues run on the GPU at the same
//! time, so the time of ops that overlap is counted for each of them, and the sum over programs
//! can exceed the wall time. The driver decides when a stamp is taken relative to the work
//! around it. Bundles get no stamps (no queries in bundles), and a recording that runs again
//! without a Reset is counted once.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use windows::core::{Interface, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// begin / end pairs in flight (a ring): two queries and 16 readback bytes each
const PAIRS: usize = 1 << 17;
/// programs a log names
const TOP: usize = 15;
const POLL: Duration = Duration::from_millis(100);
const WINDOW: Duration = Duration::from_secs(2);
/// windows between two logs of the totals since the start
const SINCE_START_EVERY: u32 = 10;

#[cfg(test)]
thread_local! {
    /// The profile on for the current thread's calls only (`breadcrumbs::FORCE` alike).
    pub(super) static FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_GPU_PROFILE").is_ok_and(|v| v.trim() == "1"));
    #[cfg(test)]
    if FORCE.with(|f| f.get()) {
        return true;
    }
    *ON
}

/// The op a pair times.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Kind {
    Dispatch,
    /// an ExecuteIndirect of dispatches with a translated pipeline
    IndirectDispatch,
    Draw,
    /// an ExecuteIndirect of draws with a translated pipeline
    IndirectDraw,
    /// an ExecuteIndirect with a PC pipeline
    PcIndirect,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Dispatch => "Dispatch",
            Kind::IndirectDispatch => "ExecuteIndirect (dispatches)",
            Kind::Draw => "draw",
            Kind::IndirectDraw => "ExecuteIndirect (draws)",
            Kind::PcIndirect => "ExecuteIndirect with a PC pipeline",
        }
    }
}

/// A program: the kind of op and its console shaders, (CS, 0) or (VS, PS); (0, 0): none known.
type Key = (Kind, u64, u64);

struct Op {
    key: Key,
    /// a dispatch's groups, a draw's (vertices or indices, instances, 0), an ExecuteIndirect's
    /// (max count, 0, 0)
    size: [u32; 3],
}

#[derive(Default, Clone, Debug, PartialEq)]
struct Total {
    calls: u64,
    ticks: u64,
    /// ticks of the longest call, and its size
    max: u64,
    size: [u32; 3],
}

/// What a pair of readback words holds.
#[derive(Debug, PartialEq)]
enum Stamps {
    /// not both written yet
    Pending,
    /// the end before the begin: the GPU wrote an earlier op's pair over the slot
    Bad,
    Span(u64),
}

fn stamps(begin: u64, end: u64) -> Stamps {
    match (begin, end) {
        (0, _) | (_, 0) => Stamps::Pending,
        (b, e) if e < b => Stamps::Bad,
        (b, e) => Stamps::Span(e - b),
    }
}

fn add(totals: &mut HashMap<Key, Total>, key: Key, ticks: u64, size: [u32; 3]) {
    let t = totals.entry(key).or_default();
    t.calls += 1;
    t.ticks += ticks;
    if ticks >= t.max {
        t.max = ticks;
        t.size = size;
    }
}

/// The `n` programs of `totals` with the most ticks, most first.
fn top(totals: &HashMap<Key, Total>, n: usize) -> Vec<(Key, Total)> {
    let mut v: Vec<(Key, Total)> = totals.iter().map(|(k, t)| (*k, t.clone())).collect();
    v.sort_by(|a, b| b.1.ticks.cmp(&a.1.ticks).then(b.1.calls.cmp(&a.1.calls)).then((a.0 .1, a.0 .2).cmp(&(b.0 .1, b.0 .2))));
    v.truncate(n);
    v
}

/// A console compute program as its pipeline was made.
#[derive(Clone, Copy)]
struct Shape {
    threads: [u32; 3],
    wave64: bool,
    /// run as two 32-lane halves (`docs/wave64-on-wave32.md`)
    split: bool,
    code: usize,
    dxil: usize,
}

/// console CS hash → its shape
static SHAPES: Mutex<Option<HashMap<u64, Shape>>> = Mutex::new(None);

/// A translated compute pipeline of console shader `blob` (hash `hash`, `dxil` bytes of DXIL)
/// was made, on a device that runs wave64 programs split (`split_device`).
pub(super) fn note_compute(hash: u64, blob: &[u8], dxil: usize, split_device: bool) {
    if !on() {
        return;
    }
    let mut g = SHAPES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if m.contains_key(&hash) {
        return;
    }
    if let Some((threads, wave64, code)) = crate::gpu::recomp::pso::compute_shape(blob) {
        m.insert(hash, Shape { threads, wave64, split: wave64 && split_device, code, dxil });
    }
}

/// ticks a second of the title queues' timestamps (0: no queue seen yet)
static FREQUENCY: AtomicU64 = AtomicU64::new(0);

/// `queue` (a title queue) executes lists: its timestamp frequency converts the pairs' ticks.
pub(super) unsafe fn note_queue(queue: *mut c_void) {
    if !on() || queue.is_null() || FREQUENCY.load(Ordering::Relaxed) != 0 {
        return;
    }
    const SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, GetTimestampFrequency) / size_of::<usize>();
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> HRESULT = transmute(pc_fn(queue, SLOT));
    let mut hz = 0u64;
    if f(queue, &mut hz).is_ok() && hz != 0 && FREQUENCY.swap(hz, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: XCL_GPU_PROFILE: GPU timestamps at {hz} Hz");
    }
}

struct Profile {
    _heap: ID3D12QueryHeap,
    _buf: ID3D12Resource,
    heap: usize,
    buf: usize,
    /// the readback buffer's mapping: two u64 a pair
    words: usize,
    next: usize,
    ops: Vec<Option<Op>>,
    in_flight: usize,
    window: HashMap<Key, Total>,
    since_start: HashMap<Key, Total>,
    /// pairs taken again before the GPU wrote them, and pairs the GPU wrote out of order
    lost: u64,
    bad: u64,
}

// SAFETY: the readback mapping lives as long as `_buf`; D3D12 objects are free-threaded.
unsafe impl Send for Profile {}

static STATE: Mutex<Option<Profile>> = Mutex::new(None);

unsafe fn init(dev: &ID3D12Device) -> Option<Profile> {
    let mut heap: Option<ID3D12QueryHeap> = None;
    let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2 * PAIRS as u32, NodeMask: 0 };
    if let Err(e) = dev.CreateQueryHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: XCL_GPU_PROFILE: no timestamp query heap ({e})");
        return None;
    }
    let heap = heap?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (PAIRS * 16) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut buf: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
        crate::xlog_warn!("D3D12.X: XCL_GPU_PROFILE: no readback buffer ({e})");
        return None;
    }
    let buf = buf?;
    let mut p = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    std::ptr::write_bytes(p.cast::<u8>(), 0, PAIRS * 16);
    let _ = std::thread::Builder::new().name("xcl-gpu-profile".into()).spawn(|| {
        let (mut last, mut windows) = (Instant::now(), 0u32);
        loop {
            std::thread::sleep(POLL);
            poll();
            if last.elapsed() >= WINDOW && report(last.elapsed(), (windows + 1) % SINCE_START_EVERY == 0) {
                windows += 1;
                last = Instant::now();
            }
        }
    });
    crate::xlog_info!("D3D12.X: XCL_GPU_PROFILE: draws, dispatches and ExecuteIndirects of console code are timed on the GPU; the programs that take the most GPU time are logged every 2 s");
    Some(Profile {
        heap: heap.as_raw() as usize,
        buf: buf.as_raw() as usize,
        _heap: heap,
        _buf: buf,
        words: p as usize,
        next: 0,
        ops: (0..PAIRS).map(|_| None).collect(),
        in_flight: 0,
        window: HashMap::new(),
        since_start: HashMap::new(),
        lost: 0,
        bad: 0,
    })
}

/// The profile, made on first use with `dev` (outside the lock: the device calls go through
/// XCL's hooks).
fn with_state<R>(dev: impl FnOnce() -> Option<ID3D12Device>, f: impl FnOnce(&mut Profile) -> R) -> Option<R> {
    static INIT: std::sync::Once = std::sync::Once::new();
    if !INIT.is_completed() {
        let dev = dev()?;
        INIT.call_once(|| {
            let s = unsafe { init(&dev) };
            *STATE.lock().unwrap() = s;
        });
    }
    STATE.lock().unwrap().as_mut().map(f)
}

impl Profile {
    fn word(&self, i: usize) -> &AtomicU64 {
        unsafe { &*(self.words as *const AtomicU64).add(i) }
    }
}

/// The PC method in `slot` of `obj` (an adopted object's own, not XCL's console-layout entry).
unsafe fn pc_fn(obj: *mut c_void, slot: usize) -> usize {
    if real::is_adopted(obj) {
        real::orig_fn(obj, slot)
    } else {
        *(*(obj as *const *const usize)).add(slot)
    }
}

type EndQueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);
type ResolveFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);

/// The begin stamp of an op on a list: `end` records the end stamp and the resolve.
pub(super) struct Pair {
    k: u32,
    heap: usize,
    buf: usize,
    end_query: EndQueryFn,
    resolve: ResolveFn,
}

/// Before the PC call of an op of console code on `list` (`pso`: its pipeline as bound).
pub(super) unsafe fn begin(list: *mut c_void, pso: usize, kind: Kind, size: [u32; 3]) -> Option<Pair> {
    if !on() || list.is_null() {
        return None;
    }
    const GET_TYPE: usize = offset_of!(ID3D12CommandList_Vtbl, GetType) / size_of::<usize>();
    const END_QUERY: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, EndQuery) / size_of::<usize>();
    const RESOLVE: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, ResolveQueryData) / size_of::<usize>();
    let get_type: unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE = transmute(pc_fn(list, GET_TYPE));
    let ty = get_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return None;
    }
    let (a, b) = real::pso_shaders(super::pso_async::resolve(pso)).unwrap_or((0, 0));
    let key = (kind, a, b);
    let (k, heap, buf) = with_state(
        || {
            let mut dev: Option<ID3D12Device> = None;
            ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
            dev
        },
        |s| {
            let k = s.next;
            s.next = (k + 1) % PAIRS;
            if s.ops[k].is_some() {
                s.lost += 1;
            } else {
                s.in_flight += 1;
            }
            s.word(2 * k).store(0, Ordering::Release);
            s.word(2 * k + 1).store(0, Ordering::Release);
            s.ops[k] = Some(Op { key, size });
            (k as u32, s.heap, s.buf)
        },
    )?;
    let pair = Pair { k, heap, buf, end_query: transmute(pc_fn(list, END_QUERY)), resolve: transmute(pc_fn(list, RESOLVE)) };
    (pair.end_query)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k);
    Some(pair)
}

/// After the PC call of the op `begin` gave `pair` for: `mark` (the op's breadcrumb), then the
/// second stamp and the pair's resolve. The resolve waits on the GPU for the op to finish, so a
/// breadcrumb recorded behind it would land only at the op's end: the op would read "not
/// started" while it runs (`breadcrumbs::tests::an_op_the_profile_times_reads_running_while_the_gpu_runs_it`).
pub(super) unsafe fn end(list: *mut c_void, pair: Option<Pair>, mark: impl FnOnce()) {
    mark();
    let Some(p) = pair else { return };
    (p.end_query)(list, p.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * p.k + 1);
    (p.resolve)(list, p.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * p.k, 2, p.buf, 16 * p.k as u64);
}

/// Add the pairs the GPU has written to their programs.
fn poll() {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let mut left = s.in_flight;
    for k in 0..PAIRS {
        if left == 0 {
            break;
        }
        let Some(op) = &s.ops[k] else { continue };
        left -= 1;
        let span = match stamps(s.word(2 * k).load(Ordering::Acquire), s.word(2 * k + 1).load(Ordering::Acquire)) {
            Stamps::Pending => continue,
            Stamps::Bad => None,
            Stamps::Span(t) => Some(t),
        };
        let (key, size) = (op.key, op.size);
        match span {
            Some(t) => {
                add(&mut s.window, key, t, size);
                add(&mut s.since_start, key, t, size);
            }
            None => s.bad += 1,
        }
        s.ops[k] = None;
        s.in_flight -= 1;
    }
}

fn program(key: Key, shapes: &HashMap<u64, Shape>) -> String {
    let (kind, a, b) = key;
    match kind {
        Kind::Dispatch | Kind::IndirectDispatch => {
            let shape = shapes.get(&a).map_or(String::new(), |s| {
                format!(
                    "; {}x{}x{} threads, {}, {} B code, {} B DXIL",
                    s.threads[0],
                    s.threads[1],
                    s.threads[2],
                    if s.split { "wave64 split in halves" } else if s.wave64 { "wave64" } else { "wave32" },
                    s.code,
                    s.dxil
                )
            });
            format!("CS {a:016x} {}{shape}", kind.name())
        }
        Kind::Draw | Kind::IndirectDraw if a != 0 => format!("VS {a:016x} PS {b:016x} {}", kind.name()),
        Kind::Draw | Kind::IndirectDraw => format!("{} with no console shader known", kind.name()),
        Kind::PcIndirect => kind.name().to_owned(),
    }
}

fn size_text(kind: Kind, size: [u32; 3]) -> String {
    match kind {
        Kind::Dispatch => format!("{}x{}x{} groups", size[0], size[1], size[2]),
        Kind::Draw => format!("{} vertices or indices x {} instances", size[0], size[1]),
        Kind::IndirectDispatch | Kind::IndirectDraw | Kind::PcIndirect => format!("max count {} (its counts are the GPU's)", size[0]),
    }
}

/// Calls and ticks of `totals`, summed.
fn sums(totals: &HashMap<Key, Total>) -> (u64, u64) {
    totals.values().fold((0, 0), |(n, ticks), t| (n + t.calls, ticks + t.ticks))
}

/// Log the programs that took the most GPU time in the last `elapsed` (and since the start).
/// False: nothing logged (no queue's timestamp frequency yet); the window goes on.
fn report(elapsed: Duration, since_start: bool) -> bool {
    let hz = FREQUENCY.load(Ordering::Relaxed);
    if hz == 0 {
        return false;
    }
    let (window, all, in_flight, lost, bad) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return false };
        let all = since_start.then(|| (top(&s.since_start, TOP), sums(&s.since_start)));
        (std::mem::take(&mut s.window), all, s.in_flight, s.lost, s.bad)
    };
    let ms = |ticks: u64| ticks as f64 * 1000.0 / hz as f64;
    let shapes = SHAPES.lock().unwrap().clone().unwrap_or_default();
    let lines = |what: &str, totals: Vec<(Key, Total)>, (ops, ticks): (u64, u64)| {
        crate::xlog_info!(
            "D3D12.X: XCL_GPU_PROFILE: {what}: {:.1} ms of GPU time in {ops} op(s) of console code ({in_flight} in flight, {lost} lost, {bad} out of order); the {} that took most:",
            ms(ticks),
            totals.len()
        );
        for (i, ((kind, a, b), t)) in totals.into_iter().enumerate() {
            crate::xlog_info!(
                "D3D12.X: XCL_GPU_PROFILE:   {:>2}. {:>9.1} ms  {}: {} call(s), longest {:.1} ms ({})",
                i + 1,
                ms(t.ticks),
                program((kind, a, b), &shapes),
                t.calls,
                ms(t.max),
                size_text(kind, t.size)
            );
        }
    };
    if !window.is_empty() {
        lines(&format!("last {:.1} s", elapsed.as_secs_f64()), top(&window, TOP), sums(&window));
    }
    if let Some((all, sum)) = all {
        lines("since the start", all, sum);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pair_counts_once_both_stamps_are_written_and_in_order() {
        assert_eq!(stamps(0, 0), Stamps::Pending);
        // (the resolve's copy can be seen half done)
        assert_eq!(stamps(1_000, 0), Stamps::Pending);
        assert_eq!(stamps(0, 1_000), Stamps::Pending);
        assert_eq!(stamps(2_000, 1_000), Stamps::Bad);
        assert_eq!(stamps(1_000, 1_000), Stamps::Span(0));
        assert_eq!(stamps(1_000, 4_500), Stamps::Span(3_500));
    }

    #[test]
    fn programs_are_ranked_by_their_total_and_keep_their_longest_call() {
        let mut t = HashMap::new();
        let (a, b, c) = ((Kind::Dispatch, 0xA, 0), (Kind::IndirectDispatch, 0xB, 0), (Kind::Draw, 0xC, 0xD));
        add(&mut t, a, 10, [1, 1, 1]);
        add(&mut t, a, 30, [8, 4, 1]);
        add(&mut t, a, 20, [2, 1, 1]);
        add(&mut t, b, 100, [1, 0, 0]);
        add(&mut t, c, 5, [3, 1, 0]);
        let r = top(&t, 2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], (b, Total { calls: 1, ticks: 100, max: 100, size: [1, 0, 0] }));
        assert_eq!(r[1], (a, Total { calls: 3, ticks: 60, max: 30, size: [8, 4, 1] }));
        // the same op with the same program under two kinds stays two programs
        add(&mut t, (Kind::IndirectDispatch, 0xA, 0), 1, [1, 0, 0]);
        assert_eq!(top(&t, 10).len(), 4);
    }
}
