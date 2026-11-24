//! `XCL_GPU_BREADCRUMBS=1`: where a queue that stopped is.
//!
//! Three kinds of marks, all read back by a reporter thread every second:
//! * ops: after each draw, dispatch and ExecuteIndirect of console code the PC list writes the
//!   op's id twice into a readback buffer (`WriteBufferImmediate`): MARKER_IN (the commands
//!   before it have begun: the op is with the GPU, "running") and MARKER_OUT (it has
//!   finished). Ops console code skipped are noted too, without a GPU write.
//!   "Running" is not "being worked on": while one queue's op holds the GPU, the other queues
//!   of the device still hand their next op over, and it stands "running" behind it. (Runs 9,
//!   12 and 14 of a retail title: one queue at an ExecuteIndirect "not started" with all
//!   before it done, three at the first dispatch of their list, a shader of 88 bytes with no
//!   loop, "running" for 30 s.)
//! * calls: before each title call that records GPU work (draws, dispatches, copies, clears,
//!   resolves, barriers, queries, the console's fills / writes / waits, KickoffX, Close) the
//!   same two writes go to the list (`title_list` calls `reached`). MARKER_OUT there means all
//!   the work recorded before the call has finished. Without them the work between two ops
//!   (copies, clears, barriers, what XCL records inside a call) had no mark, and a queue
//!   stuck in it was reported at the next op as "not started".
//!   A list keeps its marks until it is reset: one submitted again (`before_execute`) has
//!   them cleared first where its run before has finished, so they show this run.
//! * queue events: each title queue gets a fence of XCL's own, signalled after every chunk of
//!   lists it executes and after every wait it is given (the title's queue Wait / Signal, the
//!   Wait64 gates and bridged GPU waits, the sparse and VRAM page fences). Its completed value
//!   is the last event the queue has passed, whatever ran in between.
//!
//! A queue whose fence has not moved for 2 s with events still queued is logged once: the
//! event it is in, the one it passed before, what is queued after it and, for executed lists,
//! the first mark not passed and the call or op the GPU is in (shaders by their
//! `XCL_DUMP_SHADERS` dump names; an ExecuteIndirect there with its arguments, group or draw
//! counts the GPU made: as the CPU sees them and as the GPU reads them now, in their buffer
//! and in the memory window). With those reports, the marks on any list whose start the GPU
//! wrote and whose end it did not are logged: what it works on, whatever the queue (none: it
//! is busy with work that is not an op of console code). A logged queue that moves on again
//! is logged too (a long op, not a hang). `XCL_STALL_REMOVE_S=N`: once a queue has stood still
//! N s (at least the 3 s of its report), every queue's lists in flight are logged with what
//! each got to by its marks, then the device is removed and the run ends. (DRED, `XCL_DRED=1`,
//! reports no command list after a removal by `RemoveDevice`, even one held behind a queue
//! wait: `real::dred_tests`.)

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

#[cfg(test)]
thread_local! {
    /// Breadcrumbs on for the current thread's calls only, so a test that turns them on (and
    /// one that fails with them on) does not turn them on for other tests' fake lists.
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
    /// where an indirect op's arguments are in unified memory (0: not there, or no such op)
    args: usize,
    /// the PC buffer an indirect op takes its arguments from and their offset in it (0: no
    /// such op)
    args_buf: (usize, u64),
}

struct Event {
    n: u64,
    what: String,
    /// the PC lists it executed, in order (none for a wait or signal)
    lists: Vec<usize>,
    /// ops of the lists this event executed, in recording order
    ops: Vec<u32>,
    /// (list, times) of those whose recording had run before
    again: Vec<(usize, u32)>,
}

/// The marks of the recording a PC list holds.
#[derive(Default)]
struct Recording {
    ops: Vec<u32>,
    /// times it went to a queue
    submitted: u32,
    /// (queue, its event) of the last time
    last: Option<(usize, u64)>,
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
    /// PC list → the marks of its recording (until it is reset: it may go to a queue again)
    recordings: HashMap<usize, Recording>,
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
        recordings: HashMap::new(),
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

    /// Mark `id` as a report shows it.
    fn view(&self, id: u32) -> OpView {
        match self.entry(id) {
            Some(e) => OpView { id, what: e.what.clone(), pso: e.pso, state: self.state(e), call: e.call, list: e.list, args: e.args, args_buf: e.args_buf, args_ref: None },
            None => OpView { id, what: "(overwritten)".to_owned(), pso: 0, state: "unknown", call: false, list: 0, args: 0, args_buf: (0, 0), args_ref: None },
        }
    }
}

/// After the PC call of an op of console code on `list` (`pso`: its pipeline as bound).
pub(super) unsafe fn after(list: *mut c_void, pso: usize, what: impl FnOnce() -> String) {
    record(list, pso, what, true);
}

/// `after` for an ExecuteIndirect that takes its arguments from `buf` (a PC buffer and the
/// offset in it), at `args` in unified memory (0: not there): a report of the op not finished
/// says what they hold then – counts the GPU made, which no line of the recording has.
pub(super) unsafe fn after_indirect(list: *mut c_void, pso: usize, args: usize, buf: (usize, u64), what: impl FnOnce() -> String) {
    let Some(id) = record(list, pso, what, true) else { return };
    let _ = with_state(|| None, |s| {
        if let Some(e) = s.entries[id as usize % SLOTS].as_mut().filter(|e| e.id == id) {
            e.args = args;
            e.args_buf = buf;
        }
    });
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
        9 => "Close",
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

unsafe fn record(list: *mut c_void, pso: usize, what: impl FnOnce() -> String, gpu: bool) -> Option<u32> {
    if !on() || list.is_null() {
        return None;
    }
    let real_pso = super::pso_async::resolve(pso);
    let stand_in = if super::pso_async::is_pending(pso) { " [stand-in: pipeline still being built]" } else { "" };
    push(list, real_pso, format!("{}{stand_in}", what()), gpu, false)
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
        s.entries[slot] = Some(Entry { id, pso, what, gpu, call, list: list as usize, args: 0, args_buf: (0, 0) });
        let r = s.recordings.entry(list as usize).or_default();
        if r.submitted > 0 || r.ops.len() >= SLOTS {
            // a new recording after a reset `list_reset` was not told of (KickoffX, a spare
            // segment taken again)
            *r = Recording::default();
        }
        r.ops.push(id);
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
    let again = with_state(|| None, |s| {
        let line = s.queues.entry(queue as usize).or_insert_with(|| new_line.expect("queue line made above"));
        line.next += 1;
        if (line.signal)(queue, line.fence.as_raw(), line.next).is_err() {
            line.next -= 1;
            return Vec::new();
        }
        let n = line.next;
        let (mut ops, mut again) = (Vec::new(), Vec::new());
        for l in lists {
            if let Some(r) = s.recordings.get_mut(&(*l as usize)) {
                ops.extend_from_slice(&r.ops);
                r.submitted += 1;
                r.last = Some((queue as usize, n));
                if r.submitted > 1 {
                    again.push((*l as usize, r.submitted));
                }
            }
        }
        // (what a recording that runs again holds, for its line below)
        let held: Vec<(usize, u32, usize, Vec<OpView>)> = again
            .iter()
            .map(|&(l, times)| {
                let ops = &s.recordings[&l].ops;
                (l, times, ops.len(), ops.iter().map(|&id| s.view(id)).filter(|o| !o.call).take(SHOWN).collect())
            })
            .collect();
        let line = s.queues.get_mut(&(queue as usize)).expect("queue line");
        if line.events.len() >= EVENTS {
            line.events.pop_front();
        }
        line.events.push_back(Event { n, what, lists: lists.iter().map(|&l| l as usize).collect(), ops, again });
        held
    });
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    for (list, times, marks, ops) in again.unwrap_or_default() {
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
            let ops: Vec<String> = ops.iter().map(OpView::name).collect();
            crate::xlog_warn!(
                "D3D12.X: breadcrumbs: list {list:#x} runs again on {} without a Reset (time {times}): the {marks} mark(s) of its recording stand for this run now; its first ops: [{}]",
                super::list_x::queue_label(queue as usize),
                ops.join("; ")
            );
        }
    }
}

/// `list` starts a new recording (Reset), or is gone: the marks of the one before are no
/// longer its own. (Without it the Close of a list closed when it was made, and never run,
/// stood first among the marks of its first recording, never passed.)
pub(super) fn list_reset(list: usize) {
    if !on() {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.recordings.remove(&list);
    }
}

/// `lists` are about to go to a queue (call before its ExecuteCommandLists): the marks of a
/// recording that ran before are cleared, so that what they show is this run. Not while its
/// run before is still on a queue: they stay that run's then.
pub(super) unsafe fn before_execute(lists: &[*mut c_void]) {
    if !on() {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        for l in lists {
            let Some(r) = s.recordings.get(&(*l as usize)).filter(|r| r.submitted > 0) else { continue };
            if r.last.is_some_and(|(q, n)| s.queues.get(&q).is_some_and(|line| line.fence.GetCompletedValue() < n)) {
                continue;
            }
            for &id in &r.ops {
                if s.entry(id).is_some() {
                    for k in 0..2 {
                        (*(s.words as *const AtomicU32).add(2 * (id as usize % SLOTS) + k)).store(0, Ordering::Release);
                    }
                }
            }
        }
    }
}

/// What the title call just marked on `list` (`reached`) does that its name does not say (a
/// copy's size), for the mark.
pub(super) fn detail(list: *mut c_void, what: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        let Some(&id) = s.recordings.get(&(list as usize)).and_then(|r| r.ops.last()) else { return };
        // (a call mark's text ends with its console slot until it has its detail)
        if let Some(e) = s.entries[id as usize % SLOTS].as_mut().filter(|e| e.id == id && e.call && e.what.ends_with(')')) {
            e.what.push_str(": ");
            e.what.push_str(&what());
        }
    }
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
    args: usize,
    args_buf: (usize, u64),
    /// a reference on `args_buf`'s buffer, for the mark whose line reads it on the GPU
    /// (`hold_arguments`)
    args_ref: Option<ID3D12Resource>,
}

/// For the marks whose line reads an indirect op's arguments on the GPU (`OpView::text_at`:
/// the first mark not passed of each list), a reference on the buffer of the arguments. Taken
/// with the marks, under the lock: the title can release the buffer once the queue has moved
/// on, and the read comes after the lock, on a queue no late release waits for. Not kept from
/// the recording on: a reference there would be the last one at the title's Release.
unsafe fn hold_arguments(ops: &mut [OpView]) {
    let mut lists: Vec<usize> = Vec::new();
    for o in ops.iter_mut().filter(|o| !finished(o.state)) {
        if lists.contains(&o.list) {
            continue;
        }
        lists.push(o.list);
        if !o.call && o.args_buf.0 != 0 {
            o.args_ref = ID3D12Resource::from_raw_borrowed(&(o.args_buf.0 as *mut c_void)).cloned();
        }
    }
}

impl OpView {
    /// The mark without its state.
    fn name(&self) -> String {
        if self.call {
            format!("#{} {}", self.id, self.what)
        } else {
            format!("#{} {} ({})", self.id, self.what, shaders(self.pso))
        }
    }

    fn text(&self) -> String {
        if self.call {
            format!("#{} {} ({})", self.id, self.what, self.state)
        } else if self.args != 0 && !finished(self.state) {
            format!("#{} {} ({}, {}; {})", self.id, self.what, shaders(self.pso), self.state, arguments_now(self.args))
        } else {
            format!("#{} {} ({}, {})", self.id, self.what, shaders(self.pso), self.state)
        }
    }

    /// `text` of the mark a list stands at: an indirect op's also says what the GPU reads as
    /// its arguments now (`arguments_on_gpu`: a GPU copy, so only for this one mark, whose
    /// buffer `hold_arguments` keeps alive).
    fn text_at(&self) -> String {
        let mut text = self.text();
        if let Some(buf) = self.args_ref.as_ref().filter(|_| text.ends_with(')')) {
            text.pop();
            text.push_str("; ");
            text.push_str(&arguments_on_gpu((buf.as_raw() as usize, self.args_buf.1), self.args));
            text.push(')');
        }
        text
    }
}

/// The first words of an indirect op's arguments as the GPU reads them now (`real::gpu_read`):
/// from `buf`, as ExecuteIndirect takes them, and, where they are in unified memory (`at`),
/// from the memory window at the same address, as shaders wrote them. The two are one piece
/// of video memory; in a sparse window that takes the buffer's tiles being the window's
/// (`sparse::place_buffer`), and a count that differs says they are not.
fn arguments_on_gpu(buf: (usize, u64), at: usize) -> String {
    let room = unsafe { ID3D12Resource::from_raw_borrowed(&(buf.0 as *mut c_void)).map_or(0, |r| r.GetDesc().Width) }.saturating_sub(buf.1);
    let n = (room.min(20) & !3) as usize;
    if n == 0 {
        return "no GPU read of them (past the end of their buffer)".to_owned();
    }
    let window = if at != 0 { real::uma_window_buffer(at, n) } else { None };
    let parts: Vec<(usize, u64, usize)> = std::iter::once((buf.0, buf.1, n)).chain(window.map(|(b, off)| (b, off, n))).collect();
    let words = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect() };
    match unsafe { real::gpu_read(&parts) } {
        Ok(b) => gpu_words(&words(&b[..n]), window.map(|_| words(&b[n..])).as_deref()),
        Err(e) => format!("no GPU read of them ({e})"),
    }
}

/// `arguments_on_gpu`'s text for the words read from the op's buffer and from the window.
fn gpu_words(buffer: &[u32], window: Option<&[u32]>) -> String {
    let hex = |w: &[u32]| format!("[{}]", w.iter().map(|x| format!("{x:#x}")).collect::<Vec<_>>().join(", "));
    match window {
        None => format!("the GPU reads {} in their buffer", hex(buffer)),
        Some(w) if w == buffer => format!("the GPU reads {} there, in their buffer and in the memory window alike", hex(buffer)),
        Some(w) => format!("the GPU reads {} in their buffer and {} at their address in the memory window: the two are not the same video memory", hex(buffer), hex(w)),
    }
}

/// The first words of an indirect op's arguments at `at` as the CPU reads them now (a
/// dispatch's three group counts first). A coherent window is the GPU's memory too; a sparse
/// window's CPU view can be behind what the GPU wrote.
fn arguments_now(at: usize) -> String {
    if !crate::gpu::shader::readable(at, 20) {
        return format!("its arguments at {at:#x} cannot be read");
    }
    let w = unsafe { (at as *const [u32; 5]).read_unaligned() };
    let view = if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at)) { ", a sparse window's CPU view" } else { "" };
    format!("its arguments at {at:#x} now hold [{:#x}, {:#x}, {:#x}, {:#x}, {:#x}]{view}", w[0], w[1], w[2], w[3], w[4])
}

/// The work of a mark is behind the GPU: an op finished or skipped, a call mark passed.
fn finished(state: &str) -> bool {
    matches!(state, "done" | "skipped" | "passed")
}

/// Where the GPU is in `ops` (an event's marks in recording order): in the commands recorded
/// between the last mark whose work is behind it and the first one whose work is not. A call
/// mark sits before every title call that records GPU work, so after a passed call mark these
/// are that call's commands (with what XCL recorded for it). When the two marks are on
/// different lists, the GPU is between those lists: past the last mark of the one, before the
/// first of the other. None: every mark passed.
fn culprit(ops: &[OpView]) -> Option<String> {
    let i = ops.iter().position(|o| !finished(o.state))?;
    let first = &ops[i];
    // (a skipped op recorded nothing)
    Some(match ops[..i].iter().rev().find(|o| o.state != "skipped") {
        Some(p) if p.list != first.list => format!(
            "the GPU is between lists: past the last mark of list {:#x}, {}, and before the first mark of list {:#x}, {} (between them: what list {:#x} records after its last mark, any list without marks, and the start of list {:#x})",
            p.list,
            p.text(),
            first.list,
            first.text_at(),
            p.list,
            first.list
        ),
        Some(p) if p.call => format!("the GPU is in the call {} on list {:#x}: its commands; the next mark is {}", p.text(), p.list, first.text_at()),
        Some(p) => format!("the GPU is in the commands list {:#x} recorded after {} and before {} on list {:#x}", p.list, p.text(), first.text_at(), first.list),
        None => format!("the GPU is in the commands before the first mark of its lists, {} on list {:#x}", first.text_at(), first.list),
    })
}

/// What each of `lists` (an event's, in order) got to by its marks in `ops`: `list 0x…: P/M
/// passed[, at #N …]` (at: its first mark not passed), or `no marks` (no title call that
/// records GPU work, and no Close, since its Reset). `again`: (list, times) of those whose
/// recording had run before.
fn list_progress(lists: &[usize], ops: &[OpView], again: &[(usize, u32)]) -> String {
    lists
        .iter()
        .map(|&l| {
            let list = match again.iter().find(|a| a.0 == l) {
                Some(&(_, times)) => format!("list {l:#x} (run again without a Reset, time {times})"),
                None => format!("list {l:#x}"),
            };
            let marks: Vec<&OpView> = ops.iter().filter(|o| o.list == l).collect();
            if marks.is_empty() {
                return format!("{list}: no marks");
            }
            let passed = marks.iter().filter(|o| finished(o.state)).count();
            match marks.iter().find(|o| !finished(o.state)) {
                Some(o) => format!("{list}: {passed}/{} passed, at {}", marks.len(), o.text_at()),
                None => format!("{list}: {passed}/{} passed", marks.len()),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
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

/// Remove the device (`queue` has stood still `secs` s) and end the run. First, what each list
/// in flight on every queue got to by its marks (`in_flight`): DRED reports no command list
/// after `RemoveDevice` (`real::dred_lines` then says so).
fn remove_device(queue: usize, secs: u32) {
    let label = super::list_x::queue_label(queue);
    unsafe {
        let Some(dev) = device_of_queue(queue as *mut c_void) else { return };
        let Ok(dev5) = dev.cast::<ID3D12Device5>() else {
            crate::xlog_warn!("D3D12.X: breadcrumbs: XCL_STALL_REMOVE_S: no ID3D12Device5, the device stays");
            return;
        };
        crate::xlog_warn!("D3D12.X: breadcrumbs: {label} has stood still for {secs} s: the device is removed (XCL_STALL_REMOVE_S); what each list in flight got to:");
        for (q, event, lists, ops, again) in in_flight() {
            // (a wait or signal event runs no list: the queue stands at it)
            let at = if lists.is_empty() { "no list".to_owned() } else { list_progress(&lists, &ops, &again) };
            crate::xlog_warn!("D3D12.X: breadcrumbs at the removal: {} is in {event}: {at}", super::list_x::queue_label(q));
        }
        dev5.RemoveDevice();
    }
    real::stop_if_device_removed();
}

/// Each queue's event that has not passed (the one it is in), with the lists it executed and
/// their marks: (queue, event, lists, marks, lists run again).
fn in_flight() -> Vec<(usize, String, Vec<usize>, Vec<OpView>, Vec<(usize, u32)>)> {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return Vec::new() };
    let mut out = Vec::new();
    for (&q, line) in &s.queues {
        let done = unsafe { line.fence.GetCompletedValue() };
        if done >= line.next {
            continue;
        }
        let Some(cur) = line.events.iter().find(|e| e.n == done + 1) else { continue };
        let mut ops: Vec<OpView> = cur.ops.iter().map(|&id| s.view(id)).collect();
        unsafe { hold_arguments(&mut ops) };
        out.push((q, format!("#{} {}", cur.n, cur.what), cur.lists.clone(), ops, cur.again.clone()));
    }
    out
}

/// Log each queue whose fence has stood still for 2 s with events still queued, and each
/// logged one that moves on again.
fn report() {
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    // (queue, mark it stood at, mark it is at now, reports it stood)
    let mut moved: Vec<(usize, u64, u64, u32)> = Vec::new();
    let mut remove: Option<(usize, u32)> = None;
    let (stuck, running) = {
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
            let mut ops: Vec<OpView> = cur.ops.iter().map(|&id| s.view(id)).collect();
            unsafe { hold_arguments(&mut ops) };
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
                current: match cur.again.as_slice() {
                    [] => format!("#{} {}", cur.n, cur.what),
                    again => format!(
                        "#{} {} (run again without a Reset: {})",
                        cur.n,
                        cur.what,
                        again.iter().map(|&(l, times)| format!("{l:#x} time {times}")).collect::<Vec<_>>().join(", ")
                    ),
                },
                later: later.iter().take(SHOWN).map(|e| format!("#{} {}", e.n, e.what)).collect(),
                later_n: later.len(),
                ops,
                skips,
            });
        }
        // with a queue newly logged: the marks the GPU has started and not finished, on any
        // list and queue (a stuck queue's line names only its own event's marks)
        let running: Option<Vec<(OpView, Option<(usize, u64)>)>> = (!out.is_empty()).then(|| {
            started_not_finished(&s.entries, |slot, k| s.word(slot, k))
                .into_iter()
                .map(|id| {
                    let view = s.view(id);
                    let at = s.recordings.get(&view.list).and_then(|r| r.last);
                    (view, at)
                })
                .collect()
        });
        (out, running)
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
    if let Some(running) = running {
        let shown: Vec<String> = running
            .iter()
            .take(SHOWN)
            .map(|(o, at)| match at {
                Some((q, n)) => format!("{} on list {:#x} ({} event #{n})", o.text(), o.list, super::list_x::queue_label(*q)),
                None => format!("{} on list {:#x} (no queue event known)", o.text(), o.list),
            })
            .collect();
        let line = if running.is_empty() {
            "D3D12.X: breadcrumbs: no mark on any list has its start written without its end: no op of console code is running on the GPU (what keeps it busy is XCL's own work, a list without marks, or work that is not an op of console code)".to_owned()
        } else {
            format!("D3D12.X: breadcrumbs: marks the GPU started and has not finished, on any list: {} [{}]", running.len(), shown.join("; "))
        };
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

/// Ids of the marks in `entries` (slot = index, the ring of `Crumbs`) whose start the GPU wrote
/// (`word(slot, 0)`) and whose end it did not (`word(slot, 1)`): an op running, or a call mark
/// whose work before it is running. Skipped ops wrote neither.
fn started_not_finished(entries: &[Option<Entry>], word: impl Fn(usize, usize) -> u32) -> Vec<u32> {
    let mut ids: Vec<u32> = entries
        .iter()
        .enumerate()
        .filter_map(|(slot, e)| e.as_ref().filter(|e| e.gpu && word(slot, 0) == e.id && word(slot, 1) != e.id).map(|e| e.id))
        .collect();
    ids.sort_unstable();
    ids
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

            // The same recording goes to the queue again: its marks are cleared first and
            // stand for this run, and the event says the list ran before.
            queue.Wait(&gate, 2).unwrap();
            waited(queue.as_raw(), || "second test wait".into());
            before_execute(&[list.as_raw()]);
            assert_eq!((state_of(ran), state_of(call)), ("not started", "not reached"), "a recording that runs again starts with no mark passed");
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            if !real::is_adopted(queue.as_raw()) {
                executed(queue.as_raw(), &[list.as_raw()]);
            }
            let (ops, again) = {
                let g = STATE.lock().unwrap();
                let e = g.as_ref().unwrap().queues[&(queue.as_raw() as usize)].events.back().unwrap();
                (e.ops.clone(), e.again.clone())
            };
            assert_eq!((ops, again), (vec![ran, skip, call], vec![(list.as_raw() as usize, 2)]));
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(state_of(ran), "not started", "behind its wait the second run has not started");
            gate.Signal(2).unwrap();
            assert!(until(|| fence.GetCompletedValue() >= 4), "the queue did not pass the second run");
            assert_eq!((state_of(ran), state_of(call)), ("done", "passed"));

            // a Reset: the marks of the recording before are not the new one's
            list_reset(list.as_raw() as usize);
            list.Reset(&alloc, None).unwrap();
            let new = push(list.as_raw(), 0, "a new recording".into(), true, false).unwrap();
            assert_eq!(STATE.lock().unwrap().as_ref().unwrap().recordings[&(list.as_raw() as usize)].ops, vec![new]);
            // a size for a call's mark, once
            let copy = push(list.as_raw(), 0, "CopyBufferRegion (console #15)".into(), true, true).unwrap();
            detail(list.as_raw(), || "0x40 bytes".into());
            detail(list.as_raw(), || "again".into());
            assert_eq!(STATE.lock().unwrap().as_ref().unwrap().view(copy).what, "CopyBufferRegion (console #15): 0x40 bytes");
            list.Close().unwrap();
            list_reset(list.as_raw() as usize);
        }
    }

    fn mark(id: u32, state: &'static str, call: bool) -> OpView {
        OpView { id, what: if call { "CopyBufferRegion (console #15)".into() } else { "Dispatch 1x1x1".into() }, pso: 0, state, call, list: 0x10, args: 0, args_buf: (0, 0), args_ref: None }
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

    /// An ExecuteIndirect the GPU has not finished says what its arguments hold now (group
    /// counts the GPU made: a wrong one is a dispatch that does not end); a finished one does not.
    #[test]
    fn an_unfinished_indirect_op_shows_its_arguments() {
        let args = [0xFFFF_FFF0u32, 1, 1, 0, 0];
        let op = |state| OpView { args: args.as_ptr() as usize, what: "ExecuteIndirect max 1".into(), ..mark(7, state, false) };
        for state in ["not started", "running"] {
            assert!(op(state).text().ends_with(&format!("{state}; its arguments at {:#x} now hold [0xfffffff0, 0x1, 0x1, 0x0, 0x0])", args.as_ptr() as usize)), "{}", op(state).text());
        }
        assert!(!op("done").text().contains("arguments"));
        assert!(!mark(7, "running", false).text().contains("arguments"), "an op with no arguments in unified memory");
        // the mark a list stands at also asks the GPU, when it is an indirect op with a buffer
        assert_eq!(op("done").text_at(), op("done").text());
        assert_eq!(mark(7, "running", false).text_at(), mark(7, "running", false).text());
    }

    /// What the GPU reads as a stalled ExecuteIndirect's arguments is given for their buffer
    /// (what the call takes) and for the memory window at their address (what shaders wrote):
    /// two counts that differ are two pieces of video memory.
    #[test]
    fn the_gpus_reads_of_an_indirect_ops_arguments_are_compared() {
        assert_eq!(gpu_words(&[3, 1, 1], None), "the GPU reads [0x3, 0x1, 0x1] in their buffer");
        assert_eq!(gpu_words(&[0, 1, 1], Some(&[0, 1, 1])), "the GPU reads [0x0, 0x1, 0x1] there, in their buffer and in the memory window alike");
        let two = gpu_words(&[0x3F80_0000, 7, 7], Some(&[0, 1, 1]));
        assert!(two.starts_with("the GPU reads [0x3f800000, 0x7, 0x7] in their buffer and [0x0, 0x1, 0x1] at their address in the memory window: "), "{two}");
        assert!(two.ends_with("not the same video memory"), "{two}");
    }

    /// The mark a list stands at, an ExecuteIndirect not finished: its line ends with what the
    /// GPU reads in the buffer of its arguments and in the memory window (here one buffer: the
    /// same words), 12 bytes when the buffer ends there.
    #[test]
    fn the_mark_a_list_stands_at_reads_an_indirect_ops_arguments_on_the_gpu() {
        if let Err(e) = real::init() {
            return skip_test(&format!("no real D3D12.X backend: {e}"));
        }
        let u = crate::gpu::uma::get().expect("UMA");
        let Some(k) = (0..4).find(|&k| u.is_committed_window(k)) else { return skip_test("no coherent window") };
        let p = u.alloc_in(k, 20, 0).expect("20 bytes in the coherent window") as usize;
        unsafe {
            (p as *mut [u32; 5]).write_unaligned([0xFFFF_FFF0, 1, 1, 7, 9]);
            let (buf, off) = real::uma_window_buffer(p, 20).expect("the window's buffer");
            // (the buffer is held for the first mark not passed of a list, an indirect op's)
            let indirect = |id, state| OpView { args: p, args_buf: (buf, off), what: "ExecuteIndirect max 1".into(), ..mark(id, state, false) };
            let mut ops = [indirect(5, "done"), mark(6, "passed", true), indirect(7, "not started"), indirect(8, "not started"), on(0x20, mark(9, "not reached", true)), on(0x20, indirect(10, "not started"))];
            hold_arguments(&mut ops);
            assert_eq!(ops.iter().map(|o| o.args_ref.is_some()).collect::<Vec<_>>(), [false, false, true, false, false, false]);
            assert_eq!(ops[3].text_at(), ops[3].text(), "no read without the buffer held");
            let op = &ops[2];
            let text = op.text_at();
            assert!(text.starts_with(&op.text()[..op.text().len() - 1]), "{text}");
            assert!(text.ends_with("; the GPU reads [0xfffffff0, 0x1, 0x1, 0x7, 0x9] there, in their buffer and in the memory window alike)"), "{text}");
            // (no window: the buffer alone; at the end of the buffer: the words that are there)
            assert_eq!(arguments_on_gpu((buf, off), 0), "the GPU reads [0xfffffff0, 0x1, 0x1, 0x7, 0x9] in their buffer");
            let width = ID3D12Resource::from_raw_borrowed(&(buf as *mut c_void)).unwrap().GetDesc().Width;
            assert!(arguments_on_gpu((buf, width - 12), 0).matches("0x").count() == 3);
            assert_eq!(arguments_on_gpu((buf, width), 0), "no GPU read of them (past the end of their buffer)");
        }
        u.free(p as _);
    }

    fn on(list: usize, mut m: OpView) -> OpView {
        m.list = list;
        m
    }

    /// The last passed mark ends one list and the first one not passed starts the next: the GPU
    /// is put between the two lists, not in the call of the first one's last mark (run 7: q2
    /// "in the call Close" of a list whose work was done, with the next list not started).
    #[test]
    fn a_queue_past_one_lists_last_mark_and_before_the_next_lists_first_is_between_them() {
        let ops = [on(0x10, mark(1, "done", false)), on(0x10, mark(2, "passed", true)), on(0x20, mark(3, "not reached", true))];
        let at = culprit(&ops).unwrap();
        assert!(at.starts_with("the GPU is between lists: past the last mark of list 0x10, #2 "), "{at}");
        assert!(at.contains("before the first mark of list 0x20, #3 "), "{at}");
        // a skipped op at the end of the first list recorded nothing: still between the lists
        let ops = [on(0x10, mark(1, "passed", true)), on(0x10, mark(2, "skipped", false)), on(0x20, mark(3, "not started", false))];
        assert!(culprit(&ops).unwrap().starts_with("the GPU is between lists: past the last mark of list 0x10, #1 "));
    }

    /// The marks started and not finished are found on any list: an op running and a call mark
    /// whose work before it runs; not one done, one not started, a skipped op (it wrote
    /// nothing) or a slot whose words are an older mark's.
    #[test]
    fn the_marks_the_gpu_started_and_has_not_finished_are_found_on_any_list() {
        let entry = |id: u32, gpu: bool, call: bool, list: usize| Some(Entry { id, pso: 0, what: String::new(), gpu, call, list, args: 0, args_buf: (0, 0) });
        let entries = [None, entry(1, true, false, 0x10), entry(2, true, false, 0x10), entry(3, true, true, 0x20), entry(4, true, false, 0x30), entry(5, false, false, 0x30), entry(6, true, false, 0x40)];
        // (start, end) words of each slot: 1 done, 2 running, 3 reached, 4 not started, 5 skipped,
        // 6's slot still holds another mark's start
        let words: [[u32; 2]; 7] = [[0, 0], [1, 1], [2, 0], [3, 0], [0, 0], [0, 0], [9, 0]];
        assert_eq!(started_not_finished(&entries, |slot, k| words[slot][k]), vec![2, 3]);
    }

    /// At a removal each list of the event is named with what it got to.
    #[test]
    fn each_list_of_an_event_is_named_with_the_marks_it_passed() {
        let ops = [on(0x10, mark(1, "done", false)), on(0x10, mark(2, "passed", true)), on(0x20, mark(3, "not reached", true))];
        assert_eq!(
            list_progress(&[0x10, 0x20, 0x30], &ops, &[(0x20, 3)]),
            "list 0x10: 2/2 passed; list 0x20 (run again without a Reset, time 3): 0/1 passed, at #3 CopyBufferRegion (console #15) (not reached); list 0x30: no marks"
        );
    }
}
