//! `XCL_GPU_BREADCRUMBS=1`: where a queue that stopped is.
//!
//! Three kinds of marks, all read back by a reporter thread every second:
//! * ops: after each draw, dispatch and ExecuteIndirect of console code the PC list writes the
//!   op's id twice into a readback buffer (`WriteBufferImmediate`): MARKER_IN (the GPU has
//!   started it) and MARKER_OUT (it has finished). Ops console code skipped are noted too,
//!   without a GPU write.
//! * calls: before each title call that records GPU work (draws, dispatches, copies, clears,
//!   resolves, barriers, queries, the console's fills / writes / waits, KickoffX, Close) the
//!   same two writes go to the list (`title_list` calls `reached`). MARKER_OUT there means all
//!   the work recorded before the call has finished. Without them the work between two ops
//!   (copies, clears, barriers, what XCL records inside a call) had no mark, and a queue
//!   stuck in it was reported at the next op as "not started".
//! * queue events: each title queue gets a fence of XCL's own, signalled after every chunk of
//!   lists it executes and after every wait it is given (the title's queue Wait / Signal, the
//!   Wait64 gates and bridged GPU waits, the sparse and VRAM page fences). Its completed value
//!   is the last event the queue has passed, whatever ran in between.
//!
//! A queue whose fence has not moved for 2 s with events still queued is logged once: the
//! event it is in, the one it passed before, what is queued after it and, for executed lists,
//! the first mark not passed and the call or op the GPU is in (shaders by their
//! `XCL_DUMP_SHADERS` dump names). A logged queue that moves on again is logged too (a long op,
//! not a hang). `XCL_STALL_REMOVE_S=N`: once a queue has stood still N s (at least the 3 s of
//! its report), the device is removed and the run ends, so that DRED (`XCL_DRED=1`) logs the
//! operation each command list in flight got to, the commands no mark covers included.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use windows::core::{Interface, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// ring of marks; two words (started, finished) each in the readback buffer
const SLOTS: usize = 1 << 20;
const PC_WRITE_BUFFER_IMMEDIATE: usize = 66;
/// events kept per queue (older ones are dropped once the queue has passed them)
const EVENTS: usize = 1 << 14;
/// a report names at most this many events queued after the stuck one, and ops around it
const SHOWN: usize = 6;
/// reports in a row (1 s apart) a queue's fence has to stand still before it is logged
const STUCK_REPORTS: u32 = 3;
/// ops before the stuck one searched for skipped ones, on any list
const SKIP_WINDOW: u32 = 4096;

/// Breadcrumbs on for the current thread's calls only, so a test that turns them on (and one
/// that fails with them on) does not turn them on for other tests' fake lists.
#[cfg(test)]
thread_local! {
    static FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_GPU_BREADCRUMBS").is_ok_and(|v| v.trim() == "1"));
    #[cfg(test)]
    if FORCE.with(|f| f.get()) {
        return true;
    }
    *ON
}

struct Entry {
    id: u32,
    pso: usize,
    what: String,
    /// written by the GPU (false: skipped by console code)
    gpu: bool,
    /// a call mark, written before the call's commands (false: an op's, after them)
    call: bool,
    /// the PC list it is on
    list: usize,
}

struct Event {
    n: u64,
    what: String,
    /// ops of the lists this event executed, in recording order
    ops: Vec<u32>,
}

type QueueOp = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;

struct QueueLine {
    fence: ID3D12Fence,
    signal: QueueOp,
    next: u64,
    events: VecDeque<Event>,
    /// (completed value, reports it has stood there, logged)
    watch: (u64, u32, bool),
}

struct Crumbs {
    _buf: ID3D12Resource,
    gpu_va: u64,
    words: usize,
    next: u32,
    entries: Vec<Option<Entry>>,
    /// PC list → ops recorded on it since it last went to a queue
    pending: HashMap<usize, Vec<u32>>,
    queues: HashMap<usize, QueueLine>,
}

// SAFETY: the readback mapping lives as long as `_buf`; D3D12 objects are free-threaded and
// the queue Signal entries are plain function pointers.
unsafe impl Send for Crumbs {}

static STATE: Mutex<Option<Crumbs>> = Mutex::new(None);

unsafe fn init(dev: &ID3D12Device) -> Option<Crumbs> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (SLOTS * 8) as u64,
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
        crate::xlog_warn!("D3D12.X: XCL_GPU_BREADCRUMBS: no readback buffer ({e})");
        return None;
    }
    let buf = buf?;
    let mut p = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    std::ptr::write_bytes(p.cast::<u8>(), 0, SLOTS * 8);
    let _ = std::thread::Builder::new().name("xcl-breadcrumbs".into()).spawn(|| loop {
        std::thread::sleep(Duration::from_secs(1));
        report();
    });
    crate::xlog_info!("D3D12.X: XCL_GPU_BREADCRUMBS: draws and dispatches of console code mark their start and end, title queues mark each submission and wait; a queue standing still for 2 s is logged");
    Some(Crumbs {
        gpu_va: buf.GetGPUVirtualAddress(),
        _buf: buf,
        words: p as usize,
        next: 0,
        entries: (0..SLOTS).map(|_| None).collect(),
        pending: HashMap::new(),
        queues: HashMap::new(),
    })
}

/// The breadcrumb state, made on first use with `dev` (outside the lock: the device calls go
/// through XCL's hooks).
fn with_state<R>(dev: impl FnOnce() -> Option<ID3D12Device>, f: impl FnOnce(&mut Crumbs) -> R) -> Option<R> {
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

impl Crumbs {
    fn word(&self, slot: usize, k: usize) -> u32 {
        unsafe { (*(self.words as *const AtomicU32).add(2 * slot + k)).load(Ordering::Acquire) }
    }

    fn entry(&self, id: u32) -> Option<&Entry> {
        self.entries[id as usize % SLOTS].as_ref().filter(|e| e.id == id)
    }

    fn state(&self, e: &Entry) -> &'static str {
        let slot = e.id as usize % SLOTS;
        match (e.gpu, e.call, self.word(slot, 0) == e.id, self.word(slot, 1) == e.id) {
            (false, ..) => "skipped",
            (true, false, _, true) => "done",
            (true, false, true, false) => "running",
            (true, false, false, false) => "not started",
            (true, true, _, true) => "passed",
            (true, true, true, false) => "reached, the work before it still running",
            (true, true, false, false) => "not reached",
        }
    }
}

/// After the PC call of an op of console code on `list` (`pso`: its pipeline as bound).
pub(super) unsafe fn after(list: *mut c_void, pso: usize, what: impl FnOnce() -> String) {
    record(list, pso, what, true);
}

/// An op console code skipped (nothing went to the PC list).
pub(super) unsafe fn skipped(list: *mut c_void, pso: usize, what: impl FnOnce() -> String) {
    record(list, pso, what, false);
}

/// Before a title call on `list` that records GPU work (console slot `slot`, `title_list`): a
/// call mark, so that the work between two marks is one call's.
pub(super) unsafe fn reached(list: *mut c_void, slot: usize) {
    if !on() || list.is_null() {
        return;
    }
    // (a bundle runs inside another list: its marks would be its own lists' once per run)
    if ID3D12GraphicsCommandList::from_raw_borrowed(&list).is_none_or(|l| l.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
        return;
    }
    push(list, 0, format!("{} (console #{slot})", call_name(slot)), true, true);
}

/// The PC method behind the title's console list slot `slot`, as XCL dispatches it.
fn call_name(slot: usize) -> &'static str {
    match real::older_list_slot(slot) {
        9 => "Close (XCL's commands at Close follow)",
        11 => "ClearState",
        12 => "DrawInstanced",
        13 => "DrawIndexedInstanced",
        14 => "Dispatch",
        15 => "CopyBufferRegion",
        16 => "CopyTextureRegion",
        17 => "CopyResource",
        18 => "CopyTiles",
        19 => "ResolveSubresource",
        26 => "ResourceBarrier",
        27 => "ExecuteBundle",
        28 => "BeginQuery",
        29 => "EndQuery",
        30 => "ResolveQueryData",
        51 => "ClearDepthStencilView",
        52 => "ClearRenderTargetView",
        53 => "ClearUnorderedAccessViewUint",
        54 => "ClearUnorderedAccessViewFloat",
        55 => "DiscardResource",
        59 => "ExecuteIndirect",
        60 => "CopyMemoryToMemoryX",
        61 => "FillMemoryWith32BitValueX",
        62 => "FillMemoryWith64BitValueX",
        63 => "FillMemoryWith128BitValueX",
        69 => "Write32BitValueBottomOfPipeX",
        70 => "Write64BitValueBottomOfPipeX",
        71 => "Wait32BitValueX",
        72 => "Wait64BitValueX",
        78 => "DispatchX",
        81 => "DrawIndexedX",
        83 => "CopyTextureRegionX",
        84 => "CopyResourceX",
        85 => "SetPredicationBufferX",
        88 => "Write32BitValueTopOfPipeX",
        89 => "Write64BitValueTopOfPipeX",
        90 => "Write32BitValueBottomOfShaderX",
        93 => "KickoffX",
        104 => "AtomicCopyBufferUINT",
        105 => "AtomicCopyBufferUINT64",
        108 => "ResolveSubresourceRegion",
        146 => "UAV barrier",
        _ => "console method",
    }
}

unsafe fn record(list: *mut c_void, pso: usize, what: impl FnOnce() -> String, gpu: bool) {
    if !on() || list.is_null() {
        return;
    }
    let real_pso = super::pso_async::resolve(pso);
    let stand_in = if super::pso_async::is_pending(pso) { " [stand-in: pipeline still being built]" } else { "" };
    push(list, real_pso, format!("{}{stand_in}", what()), gpu, false);
}

unsafe fn device_of_list(list: *mut c_void) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
    dev
}

unsafe fn device_of_queue(queue: *mut c_void) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12CommandQueue::from_raw_borrowed(&queue)?.GetDevice(&mut dev).ok()?;
    dev
}

/// Note an op or a call mark (`call`) on `list` and, for one that goes to the GPU, record its
/// two marker writes there.
unsafe fn push(list: *mut c_void, pso: usize, what: String, gpu: bool, call: bool) -> Option<u32> {
    let (id, va) = with_state(|| device_of_list(list), |s| {
        s.next = s.next.wrapping_add(1).max(1);
        let id = s.next;
        let slot = id as usize % SLOTS;
        for k in 0..2 {
            (*(s.words as *const AtomicU32).add(2 * slot + k)).store(0, Ordering::Release);
        }
        s.entries[slot] = Some(Entry { id, pso, what, gpu, call, list: list as usize });
        let ops = s.pending.entry(list as usize).or_default();
        if ops.len() >= SLOTS {
            // a list recorded and reset without ever being executed
            ops.clear();
        }
        ops.push(id);
        (id, s.gpu_va + 8 * slot as u64)
    })?;
    if !gpu {
        return Some(id);
    }
    let params = [
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: va, Value: id },
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: va + 4, Value: id },
    ];
    let modes = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT];
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    let f: F = if real::is_adopted(list) {
        transmute(real::orig_fn(list, PC_WRITE_BUFFER_IMMEDIATE))
    } else {
        transmute(*(*(list as *const *const usize)).add(PC_WRITE_BUFFER_IMMEDIATE))
    };
    f(list, 2, params.as_ptr(), modes.as_ptr());
    Some(id)
}

unsafe fn queue_line(queue: *mut c_void, dev: &ID3D12Device) -> Option<QueueLine> {
    const SIGNAL_SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, Signal) / size_of::<usize>();
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    // the PC Signal: the console layout dispatch would take XCL's fence for a title fence
    let signal: QueueOp = if real::is_adopted(queue) { transmute(real::orig_fn(queue, SIGNAL_SLOT)) } else { q.vtable().Signal };
    Some(QueueLine { fence, signal, next: 0, events: VecDeque::new(), watch: (0, 0, false) })
}

/// Mark on `queue` that the work submitted to it so far is done: `what` was the last of it.
/// `lists`: the PC lists it executed, if it was an ExecuteCommandLists.
unsafe fn mark(queue: *mut c_void, what: impl FnOnce() -> String, lists: &[*mut c_void]) {
    if !on() || queue.is_null() {
        return;
    }
    let Some(dev) = device_of_queue(queue) else { return };
    let what = what();
    // a new queue's fence is made outside the lock (CreateFence goes through XCL's hooks)
    let known = with_state(|| Some(dev.clone()), |s| s.queues.contains_key(&(queue as usize)));
    let new_line = match known {
        None => return,
        Some(true) => None,
        Some(false) => match queue_line(queue, &dev) {
            Some(l) => Some(l),
            None => return,
        },
    };
    let _ = with_state(|| None, |s| {
        let mut ops = Vec::new();
        for l in lists {
            if let Some(v) = s.pending.remove(&(*l as usize)) {
                ops.extend(v);
            }
        }
        let line = s.queues.entry(queue as usize).or_insert_with(|| new_line.expect("queue line made above"));
        line.next += 1;
        if (line.signal)(queue, line.fence.as_raw(), line.next).is_err() {
            line.next -= 1;
            return;
        }
        if line.events.len() >= EVENTS {
            line.events.pop_front();
        }
        line.events.push_back(Event { n: line.next, what, ops });
    });
}

/// `lists` (PC lists) went to `queue` (call after the queue's ExecuteCommandLists).
pub(super) unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    mark(queue, || format!("ExecuteCommandLists of {} list(s) [{}]", lists.len(), lists.iter().map(|&l| format!("{l:p}")).collect::<Vec<_>>().join(", ")), lists);
}

/// `queue` was given a wait (call after the queue's Wait / Signal): the mark after it passes
/// once the wait is satisfied.
pub(super) unsafe fn waited(queue: *mut c_void, what: impl FnOnce() -> String) {
    mark(queue, what, &[]);
}

thread_local! {
    static LABEL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Run `f`, a queue Wait / Signal of XCL's own that goes through the title's queue entry
/// (`real::queue_dispatch`), which then marks it named `label` instead of as the title's.
pub(super) fn labelled<R>(label: impl FnOnce() -> String, f: impl FnOnce() -> R) -> R {
    if !on() {
        return f();
    }
    let old = LABEL.with(|l| l.replace(Some(label())));
    let r = f();
    LABEL.with(|l| *l.borrow_mut() = old);
    r
}

/// A queue Signal / Wait from the title's queue entry went to the GPU: the title's, or one of
/// XCL's own (named when it ran under `labelled`).
pub(super) unsafe fn queue_call(queue: *mut c_void, what: &str, fence: *mut c_void, value: u64) {
    if !on() {
        return;
    }
    let label = LABEL.with(|l| l.borrow_mut().take());
    mark(queue, || match label {
        Some(l) => format!("{what} of fence {fence:p} at {value:#x} ({l})"),
        None => format!("{what} of fence {fence:p} at {value:#x} (the title's, or XCL's own unnamed)"),
    }, &[]);
}

fn shaders(pso: usize) -> String {
    match real::pso_shaders(pso) {
        Some((cs, 0)) => format!("shader {cs:016x}"),
        Some((vs, ps)) => format!("VS {vs:016x} PS {ps:016x}"),
        None if pso == 0 => "no pipeline".to_owned(),
        None => format!("pipeline {pso:#x}, no console shader known"),
    }
}

/// One mark of a stuck event, taken under the lock.
struct OpView {
    id: u32,
    what: String,
    pso: usize,
    state: &'static str,
    call: bool,
    list: usize,
}

impl OpView {
    fn text(&self) -> String {
        if self.call {
            format!("#{} {} ({})", self.id, self.what, self.state)
        } else {
            format!("#{} {} ({}, {})", self.id, self.what, shaders(self.pso), self.state)
        }
    }
}

/// The work of a mark is behind the GPU: an op finished or skipped, a call mark passed.
fn finished(state: &str) -> bool {
    matches!(state, "done" | "skipped" | "passed")
}

/// Where the GPU is in `ops` (an event's marks in recording order): in the commands recorded
/// between the last mark whose work is behind it and the first one whose work is not. A call
/// mark sits before every title call that records GPU work, so after a passed call mark these
/// are that call's commands (with what XCL recorded for it). None: every mark passed.
fn culprit(ops: &[OpView]) -> Option<String> {
    let i = ops.iter().position(|o| !finished(o.state))?;
    let first = &ops[i];
    // (a skipped op recorded nothing)
    Some(match ops[..i].iter().rev().find(|o| o.state != "skipped") {
        Some(p) if p.call => format!("the GPU is in the call {} on list {:#x}: its commands; the next mark is {}", p.text(), p.list, first.text()),
        Some(p) => format!("the GPU is in the commands list {:#x} recorded after {} and before {} on list {:#x}", p.list, p.text(), first.text(), first.list),
        None => format!("the GPU is in the commands before the first mark of its lists, {} on list {:#x}", first.text(), first.list),
    })
}

/// What a report says about one queue, taken under the lock and formatted after it.
struct Stuck {
    queue: usize,
    done: u64,
    passed: Option<String>,
    current: String,
    later: Vec<String>,
    later_n: usize,
    /// the marks of the current event
    ops: Vec<OpView>,
    /// (id, what, pso) of ops skipped on any list in the `SKIP_WINDOW` ops before the stuck one
    skips: Vec<(u32, String, usize)>,
}

/// `XCL_STALL_REMOVE_S`: seconds a queue stands still before the device is removed.
fn remove_after() -> Option<u32> {
    static S: LazyLock<Option<u32>> = LazyLock::new(|| std::env::var("XCL_STALL_REMOVE_S").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0));
    *S
}

/// Remove the device (`queue` has stood still `secs` s), so that DRED logs the operation each
/// command list in flight got to (`real::log_device_removed`), and end the run.
fn remove_device(queue: usize, secs: u32) {
    let label = super::list_x::queue_label(queue);
    unsafe {
        let Some(dev) = device_of_queue(queue as *mut c_void) else { return };
        let Ok(dev5) = dev.cast::<ID3D12Device5>() else {
            crate::xlog_warn!("D3D12.X: breadcrumbs: XCL_STALL_REMOVE_S: no ID3D12Device5, the device stays");
            return;
        };
        crate::xlog_warn!("D3D12.X: breadcrumbs: {label} has stood still for {secs} s: the device is removed (XCL_STALL_REMOVE_S) so that DRED logs what each list in flight got to (with XCL_DRED=1)");
        dev5.RemoveDevice();
    }
    real::stop_if_device_removed();
}

/// Log each queue whose fence has stood still for 2 s with events still queued, and each
/// logged one that moves on again.
fn report() {
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    // (queue, mark it stood at, mark it is at now, reports it stood)
    let mut moved: Vec<(usize, u64, u64, u32)> = Vec::new();
    let mut remove: Option<(usize, u32)> = None;
    let stuck: Vec<Stuck> = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        let mut out = Vec::new();
        let queues: Vec<usize> = s.queues.keys().copied().collect();
        for q in queues {
            let line = s.queues.get_mut(&q).unwrap();
            let done = unsafe { line.fence.GetCompletedValue() };
            if done == u64::MAX {
                continue; // device removed
            }
            while line.events.front().is_some_and(|e| e.n + 1 < done) {
                line.events.pop_front();
            }
            if line.watch.2 && line.watch.0 != done {
                moved.push((q, line.watch.0, done, line.watch.1));
            }
            if done >= line.next {
                line.watch = (done, 0, false);
                continue;
            }
            if line.watch.0 == done {
                line.watch.1 += 1;
            } else {
                line.watch = (done, 1, false);
            }
            // (not before its report: the report is logged first)
            if remove.is_none() && remove_after().is_some_and(|n| line.watch.1 >= n.max(STUCK_REPORTS)) {
                remove = Some((q, line.watch.1));
            }
            if line.watch.1 < STUCK_REPORTS || line.watch.2 {
                continue;
            }
            line.watch.2 = true;
            let line = &s.queues[&q];
            let passed = line.events.iter().find(|e| e.n == done).map(|e| format!("#{} {}", e.n, e.what));
            let Some(cur) = line.events.iter().find(|e| e.n == done + 1) else { continue };
            let later: Vec<&Event> = line.events.iter().filter(|e| e.n > done + 1).collect();
            let ops: Vec<OpView> = cur.ops.iter().map(|&id| match s.entry(id) {
                Some(e) => OpView { id, what: e.what.clone(), pso: e.pso, state: s.state(e), call: e.call, list: e.list },
                None => OpView { id, what: "(overwritten)".to_owned(), pso: 0, state: "unknown", call: false, list: 0 },
            }).collect();
            // ops console code skipped on any list just before the stuck one: a stand-in that
            // never wrote what the stuck op reads would be among them
            let anchor = ops.iter().find(|o| !finished(o.state)).map_or(s.next, |o| o.id);
            let skips: Vec<(u32, String, usize)> = (anchor.saturating_sub(SKIP_WINDOW)..anchor)
                .filter_map(|id| s.entry(id).filter(|e| !e.gpu).map(|e| (id, e.what.clone(), e.pso)))
                .collect();
            out.push(Stuck {
                queue: q,
                done,
                passed,
                current: format!("#{} {}", cur.n, cur.what),
                later: later.iter().take(SHOWN).map(|e| format!("#{} {}", e.n, e.what)).collect(),
                later_n: later.len(),
                ops,
                skips,
            });
        }
        out
    };
    for (q, from, to, reports) in moved {
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 256 {
            crate::xlog_warn!("D3D12.X: breadcrumbs: {} moved on after standing still about {reports} s at its mark {from} (now at {to}): a long piece of work, not a hang", super::list_x::queue_label(q));
        }
    }
    for st in stuck {
        let queue = super::list_x::queue_label(st.queue);
        let ops = if st.ops.is_empty() {
            String::new()
        } else {
            let done_n = st.ops.iter().filter(|o| o.state == "done").count();
            let skipped_n = st.ops.iter().filter(|o| o.state == "skipped").count();
            let calls_n = st.ops.iter().filter(|o| o.call).count();
            match (st.ops.iter().position(|o| !finished(o.state)), culprit(&st.ops)) {
                (Some(i), Some(at)) => {
                    let before: Vec<String> = st.ops[..i].iter().rev().take(SHOWN).rev().map(OpView::text).collect();
                    let after: Vec<String> = st.ops[i + 1..].iter().take(SHOWN).map(OpView::text).collect();
                    format!(
                        "; its lists hold {} mark(s) ({calls_n} before calls; ops: {done_n} done, {skipped_n} skipped); {at}; before it: [{}]; after it: [{}]",
                        st.ops.len(), before.join("; "), after.join("; ")
                    )
                }
                _ => format!(
                    "; all {} mark(s) of its lists passed ({skipped_n} ops skipped), so the queue is in the work after the last of them (XCL's commands at the end of a list, or the next submission's start): last {}",
                    st.ops.len(), st.ops.last().map(OpView::text).unwrap_or_default()
                ),
            }
        };
        let skips: Vec<String> = st.skips.iter().rev().take(SHOWN).map(|(id, what, pso)| format!("#{id} {what} ({})", shaders(*pso))).collect();
        let line = format!(
            "D3D12.X: breadcrumbs: {queue} has stood for 2 s at its mark {}: it is in {}{ops}; passed before it: {}; queued after it: {} more [{}]; skipped on any list in the {SKIP_WINDOW} ops before it: {} [latest: {}]",
            st.done,
            st.current,
            st.passed.as_deref().unwrap_or("(nothing kept)"),
            st.later_n,
            st.later.join("; "),
            st.skips.len(),
            skips.join("; "),
        );
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 256 {
            crate::xlog_warn!("{line}");
        }
    }
    if let Some((q, secs)) = remove {
        static REMOVED: AtomicBool = AtomicBool::new(false);
        if !REMOVED.swap(true, Ordering::AcqRel) {
            remove_device(q, secs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::runner::skip_test;
    use std::time::Instant;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    fn state_of(id: u32) -> &'static str {
        let g = STATE.lock().unwrap();
        let s = g.as_ref().unwrap();
        s.state(s.entry(id).expect("entry overwritten"))
    }

    fn until(mut done: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while !done() {
            if start.elapsed() > Duration::from_secs(5) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    #[test]
    fn breadcrumbs_mark_ops_and_the_queue_events_after_them() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            // off again however the test ends (with one test thread, the next test runs on this
            // thread)
            struct Unforce;
            impl Drop for Unforce {
                fn drop(&mut self) {
                    FORCE.with(|f| f.set(false));
                }
            }
            FORCE.with(|f| f.set(true));
            let _unforce = Unforce;
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let ran = push(list.as_raw(), 0, "first".into(), true, false).expect("no breadcrumb buffer");
            let skip = push(list.as_raw(), 0, "skipped".into(), false, false).unwrap();
            let call = push(list.as_raw(), 0, "a call".into(), true, true).unwrap();
            list.Close().unwrap();
            assert_eq!(state_of(ran), "not started");
            assert_eq!(state_of(skip), "skipped");
            assert_eq!(state_of(call), "not reached");

            // a wait nothing has satisfied yet holds the list and both marks behind it
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            waited(queue.as_raw(), || "test wait".into());
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            // a queue of a device an earlier test adopted marks its ExecuteCommandLists itself
            // (`real::q_execute_command_lists`)
            if !real::is_adopted(queue.as_raw()) {
                executed(queue.as_raw(), &[list.as_raw()]);
            }
            let (fence, next, ops) = {
                let g = STATE.lock().unwrap();
                let line = &g.as_ref().unwrap().queues[&(queue.as_raw() as usize)];
                (line.fence.clone(), line.next, line.events.back().unwrap().ops.clone())
            };
            assert_eq!(next, 2);
            assert_eq!(ops, vec![ran, skip, call], "the executed event did not take the list's ops");
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(fence.GetCompletedValue(), 0, "a mark passed a wait that was not satisfied");
            assert_eq!(state_of(ran), "not started");

            gate.Signal(1).unwrap();
            assert!(until(|| fence.GetCompletedValue() >= 2), "the queue did not pass its marks");
            assert_eq!(state_of(ran), "done");
            assert_eq!(state_of(call), "passed");
        }
    }

    fn mark(id: u32, state: &'static str, call: bool) -> OpView {
        OpView { id, what: if call { "CopyBufferRegion (console #15)".into() } else { "Dispatch 1x1x1".into() }, pso: 0, state, call, list: 0x10 }
    }

    /// The GPU is put in the call whose mark it passed last, not at the next op: a dispatch
    /// done, then a copy (its call mark passed) that has not finished, and a dispatch not
    /// started behind it is the copy.
    #[test]
    fn a_stuck_queue_is_put_in_the_call_after_the_last_passed_mark() {
        let ops = [mark(1, "passed", true), mark(2, "done", false), mark(3, "passed", true), mark(4, "not reached", true), mark(5, "not started", false)];
        let at = culprit(&ops).unwrap();
        assert!(at.starts_with("the GPU is in the call #3 CopyBufferRegion"), "{at}");
        assert!(at.contains("the next mark is #4"), "{at}");
        // skipped ops recorded nothing: the call before them
        let ops = [mark(1, "passed", true), mark(2, "skipped", false), mark(3, "not reached", true)];
        assert!(culprit(&ops).unwrap().starts_with("the GPU is in the call #1"));
        // after an op's end mark: the commands between it and the next mark
        let ops = [mark(1, "done", false), mark(2, "running", false)];
        assert!(culprit(&ops).unwrap().starts_with("the GPU is in the commands list 0x10 recorded after #1"));
        let ops = [mark(1, "not reached", true)];
        assert!(culprit(&ops).unwrap().starts_with("the GPU is in the commands before the first mark"));
        assert!(culprit(&[mark(1, "passed", true), mark(2, "done", false)]).is_none());
    }
}
