//! Queue-side gates for recorded waits on CPU-visible progress values.
//!
//! A gate the CPU never sees satisfied (its producer was skipped, never submitted, or is itself
//! held behind this queue) is opened on the CPU once the queue has stood at it for
//! `XCL_WAIT_GATE_RELEASE_MS` (default 5000; `0` keeps the queue blocked for good). The work
//! after it then runs before its producer, as with the GPU wait watchdog of bridged waits
//! (`XCL_GPU_WAIT_WATCHDOG_MS`, default 5000; `0` never releases them; `list_x`).
//! Gates armed on that queue before such a release are opened `GRACE` after the queue reaches
//! them: they wait behind the same stalled producers, and k of them must not cost k release
//! times.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute, ManuallyDrop};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Foundation::E_POINTER;
use windows::Win32::Graphics::Direct3D12::*;

use super::real;
use crate::gpu::fence::{self, FenceError};

const REACHED: u64 = 1;
const OPEN: u64 = 2;
const RETIRED: u64 = 3;
const POLL: Duration = Duration::from_millis(1);
const WARN_AFTER: Duration = Duration::from_secs(2);
const DEFAULT_RELEASE: Duration = Duration::from_secs(5);
/// How long a queue stands at a gate armed before the last release on that queue.
const GRACE: Duration = Duration::from_millis(250);

/// How long a queue stands at a gate whose value the CPU does not see before the CPU opens it
/// (`XCL_WAIT_GATE_RELEASE_MS`; `0`: never).
fn release_after() -> Option<Duration> {
    static R: LazyLock<Option<Duration>> = LazyLock::new(|| {
        match std::env::var("XCL_WAIT_GATE_RELEASE_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            Some(0) => None,
            Some(ms) => Some(Duration::from_millis(ms)),
            None => Some(DEFAULT_RELEASE),
        }
    });
    *R
}

/// How long the queue stands at a gate armed at `armed_at` before the CPU opens it: the full
/// `release_after`, or `GRACE` when a gate on that queue was opened on the CPU since then
/// (`last_release`).
fn release_time(release_after: Duration, armed_at: Instant, last_release: Option<Instant>) -> Duration {
    if last_release.is_some_and(|at| armed_at < at) { release_after.min(GRACE) } else { release_after }
}

static TX: LazyLock<Result<mpsc::Sender<Gate>, String>> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new().name("xcl-wait-gates".into()).spawn(move || monitor(rx))
        .map_err(|e| format!("starting the Wait64BitValueX monitor: {e}"))?;
    Ok(tx)
});

/// What a gate waits for, for its warnings: `true` once the queue stands at the gate.
type Describe = Box<dyn Fn(bool) -> String + Send>;

/// queue → gates armed on it that the CPU has not opened yet
static CLOSED: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// Whether `queue` has a gate the CPU has not opened yet: work submitted to it now may only run
/// after a submission the title makes later, so nothing in the title's call may wait for it.
pub(super) fn closed_on(queue: usize) -> bool {
    CLOSED.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&queue))
}

/// Counts its gate in `CLOSED` until it is dropped (the gate opened or given up).
struct Closed(usize);

impl Closed {
    fn new(queue: usize) -> Self {
        *CLOSED.lock().unwrap().get_or_insert_with(HashMap::new).entry(queue).or_insert(0) += 1;
        Closed(queue)
    }
}

impl Drop for Closed {
    fn drop(&mut self) {
        let mut g = CLOSED.lock().unwrap();
        let Some(m) = g.as_mut() else { return };
        if let Some(n) = m.get_mut(&self.0) {
            *n -= 1;
            if *n == 0 {
                m.remove(&self.0);
            }
        }
    }
}

struct Gate {
    queue: usize,
    /// arm order, to count the gates queued earlier on the same queue
    seq: u64,
    device: ID3D12Device,
    // D3D12 can still be using this fence until the queue passes the retirement signal.
    // Only release it on confirmed retirement/removal, even if the worker exits unexpectedly.
    fence: ManuallyDrop<ID3D12Fence>,
    ready: Option<Box<dyn Fn() -> bool + Send>>,
    /// while the CPU has not opened the gate (`closed_on`)
    closed: Option<Closed>,
    describe: Describe,
    warn_at: Option<Instant>,
    armed_at: Instant,
    /// when the monitor first saw the queue at the gate
    reached_at: Option<Instant>,
    release_after: Option<Duration>,
    retire_at: Option<u64>,
}

enum Poll {
    Pending,
    Retired,
    KeepFence,
}

impl Gate {
    fn failed(&self, error: FenceError) -> Poll {
        crate::xlog_error!("D3D12.X: Wait64BitValueX gate on queue {:#x}: {error}", self.queue);
        if matches!(error, FenceError::Removed(_)) {
            real::stop_if_device_removed();
            Poll::Retired
        } else {
            // The queue may still refer to the fence even though we cannot service it.
            Poll::KeepFence
        }
    }

    fn warn_due(&self) -> bool {
        self.ready.is_some() && self.warn_at.is_some_and(|at| Instant::now() >= at)
    }

    /// `earlier`: gates armed before this one on the same queue and not opened yet (counted
    /// only when a warning is due). `released`: queue → when the CPU last opened a gate on it.
    fn poll(&mut self, earlier: usize, released: &mut HashMap<usize, Instant>) -> Poll {
        let value = match unsafe { fence::completed_value(&self.fence, &self.device) } {
            Ok(value) => value,
            Err(e) => return self.failed(e),
        };
        if self.retire_at.is_some_and(|target| value >= target) {
            return Poll::Retired;
        }
        let Some(ready) = &self.ready else { return Poll::Pending };
        let reached = value >= REACHED;
        if reached && self.reached_at.is_none() {
            self.reached_at = Some(Instant::now());
        }
        if !reached || !ready() {
            let held = self.reached_at.map(|at| at.elapsed());
            let release = self.release_after.map(|after| release_time(after, self.armed_at, released.get(&self.queue).copied()));
            if let (Some(held), Some(after)) = (held, release) {
                if held >= after {
                    static RELEASED: AtomicU64 = AtomicU64::new(0);
                    let n = RELEASED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 32 || n.is_power_of_two() {
                        let behind = if Some(after) == self.release_after { "" } else { "; armed before a gate on this queue was opened on the CPU" };
                        crate::xlog_warn!("D3D12.X: Wait64BitValueX gate on queue {:#x} opened on the CPU after {} ms at the gate without its value ({}{behind}) – the work after it runs before its producer ({n} so far)", self.queue, held.as_millis(), (self.describe)(true));
                    }
                    released.insert(self.queue, Instant::now());
                    return self.open();
                }
            }
            if self.warn_at.is_some_and(|at| Instant::now() >= at) {
                self.warn_at = None;
                static WARNED: AtomicU64 = AtomicU64::new(0);
                let n = WARNED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 64 || n.is_power_of_two() {
                    let waiting_on = if reached { "the memory condition" } else { "preceding queue work" };
                    let earlier = if earlier > 0 { format!("; {earlier} earlier gate(s) on this queue still closed") } else { String::new() };
                    let action = match release {
                        None => "leaving the queue blocked".to_owned(),
                        Some(after) => format!("the CPU opens it {} ms after the queue reaches it", after.as_millis()),
                    };
                    crate::xlog_warn!("D3D12.X: Wait64BitValueX gate on queue {:#x} still waiting on {waiting_on} after {} s ({}{earlier}); {action}", self.queue, WARN_AFTER.as_secs(), (self.describe)(reached));
                }
            }
            return Poll::Pending;
        }
        self.open()
    }

    fn open(&mut self) -> Poll {
        if let Err(e) = unsafe { self.fence.Signal(OPEN) } {
            return self.failed(unsafe { fence::api_error(&self.device, "ID3D12Fence::Signal", e) });
        }
        self.ready = None;
        self.closed = None;
        if self.retire_at.is_some() { Poll::Pending } else { Poll::KeepFence }
    }
}

fn monitor(rx: mpsc::Receiver<Gate>) {
    let mut pending: Vec<Gate> = Vec::new();
    // queue → when the CPU last opened a gate on it without its value
    let mut released: HashMap<usize, Instant> = HashMap::new();
    loop {
        if pending.is_empty() {
            let Ok(gate) = rx.recv() else { return };
            pending.push(gate);
        }
        pending.extend(rx.try_iter());
        let mut i = 0;
        while i < pending.len() {
            let earlier = if pending[i].warn_due() {
                let (queue, seq) = (pending[i].queue, pending[i].seq);
                pending.iter().filter(|g| g.queue == queue && g.seq < seq && g.ready.is_some()).count()
            } else {
                0
            };
            match pending[i].poll(earlier, &mut released) {
                Poll::Pending => i += 1,
                Poll::Retired => drop(ManuallyDrop::into_inner(pending.swap_remove(i).fence)),
                Poll::KeepFence => { pending.swap_remove(i); }
            }
        }
        if !pending.is_empty() {
            std::thread::sleep(POLL);
        }
    }
}

unsafe fn private_fence(device: &ID3D12Device) -> windows::core::Result<ID3D12Fence> {
    if !real::is_adopted(device.as_raw()) {
        return device.CreateFence(0, D3D12_FENCE_FLAG_NONE);
    }
    // The title's CreateFence hook keeps its fences and signals them at shutdown. Neither
    // applies to our private gates.
    const SLOT: usize = offset_of!(ID3D12Device_Vtbl, CreateFence) / size_of::<usize>();
    type Create = unsafe extern "system" fn(*mut c_void, u64, D3D12_FENCE_FLAGS, *const GUID, *mut *mut c_void) -> HRESULT;
    let create: Create = transmute(real::orig_fn(device.as_raw(), SLOT));
    let mut raw = std::ptr::null_mut();
    create(device.as_raw(), 0, D3D12_FENCE_FLAG_NONE, &ID3D12Fence::IID, &mut raw).ok()?;
    if raw.is_null() {
        return Err(E_POINTER.into());
    }
    Ok(ID3D12Fence::from_raw(raw))
}

/// Insert a GPU queue wait at this submission point. After the queue reaches it, the monitor
/// opens it when `ready` observes the requested value, or after `XCL_WAIT_GATE_RELEASE_MS` at
/// the gate; the submitter is free to submit the producer on another queue. `describe` says
/// what the gate waits for in its warnings.
///
/// `queue` must be a live native D3D12 command queue. `ready` and `describe` must be safe to
/// call on the monitor.
pub(super) unsafe fn arm(
    queue: *mut c_void, ready: impl Fn() -> bool + Send + 'static, describe: impl Fn(bool) -> String + Send + 'static,
) -> Result<(), String> {
    arm_with(queue, ready, Box::new(describe), release_after())
}

unsafe fn arm_with(
    queue: *mut c_void, ready: impl Fn() -> bool + Send + 'static, describe: Describe, release_after: Option<Duration>,
) -> Result<(), String> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue).ok_or("missing D3D12 command queue")?;
    let mut device: Option<ID3D12Device> = None;
    q.GetDevice(&mut device).map_err(|e| format!("GetDevice: {e}"))?;
    let device = device.ok_or("GetDevice returned no D3D12 device")?;
    let fence = private_fence(&device).map_err(|e| fence::api_error(&device, "CreateFence", e).to_string())?;
    let tx = TX.as_ref().map_err(Clone::clone)?;

    // Bypass the console layout dispatch: the queue slots are PC slots and the fence is not
    // registered as a title fence.
    const WAIT_SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, Wait) / size_of::<usize>();
    const SIGNAL_SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, Signal) / size_of::<usize>();
    type QueueOp = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
    let adopted = real::is_adopted(queue);
    let pc = |slot, native: QueueOp| -> QueueOp {
        if adopted { transmute(real::orig_fn(queue, slot)) } else { native }
    };
    let signal = pc(SIGNAL_SLOT, q.vtable().Signal);
    signal(queue, fence.as_raw(), REACHED).ok()
        .map_err(|e| fence::api_error(&device, "ID3D12CommandQueue::Signal (prefix)", e).to_string())?;
    let wait = pc(WAIT_SLOT, q.vtable().Wait)(queue, fence.as_raw(), OPEN).ok();
    let armed = wait.is_ok();
    let retire_at = if !armed {
        // The prefix marker was still queued and must finish before the fence is released.
        Some(REACHED)
    } else {
        // The queue cannot signal RETIRED until it consumes the wait. If the signal fails to
        // queue, still service the committed wait and keep the fence alive.
        match signal(queue, fence.as_raw(), RETIRED).ok() {
            Ok(()) => Some(RETIRED),
            Err(e) => {
                let e = fence::api_error(&device, "ID3D12CommandQueue::Signal (retirement)", e);
                crate::xlog_error!("D3D12.X: could not queue retirement for Wait64BitValueX gate on queue {queue:p}: {e}");
                None
            }
        }
    };
    let wait = wait.map_err(|e| fence::api_error(&device, "ID3D12CommandQueue::Wait", e).to_string());
    tx.send(Gate {
        queue: queue as usize,
        seq: SEQ.fetch_add(1, Ordering::Relaxed),
        device,
        fence: ManuallyDrop::new(fence),
        ready: armed.then(|| Box::new(ready) as Box<dyn Fn() -> bool + Send>),
        closed: armed.then(|| Closed::new(queue as usize)),
        describe,
        warn_at: Some(Instant::now() + WARN_AFTER),
        armed_at: Instant::now(),
        reached_at: None,
        release_after,
        retire_at,
    }).map_err(|_| "Wait64BitValueX monitor stopped after submitting to the queue; cannot service the gate".to_owned())?;
    wait
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::runner::skip_test;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    fn until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while !done() {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    unsafe fn device_and_queue() -> Option<(ID3D12Device, ID3D12CommandQueue)> {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            skip_test("no GPU");
            return None;
        }
        let dev = dev.unwrap();
        let queue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
        Some((dev, queue))
    }

    #[test]
    fn gate_without_its_value_opens_after_the_release_time_at_the_gate() {
        const RELEASE: Duration = Duration::from_millis(400);
        unsafe {
            let Some((dev, queue)) = device_and_queue() else { return };
            let new_fence = || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (ahead, tail) = (new_fence(), new_fence());
            let reached = |f: &ID3D12Fence| fence::completed_value(f, &dev).is_ok_and(|v| v >= 1);
            // Earlier work holds the queue before the gate: the release clock must not run yet.
            queue.Wait(&ahead, 1).unwrap();
            arm_with(queue.as_raw(), || false, Box::new(|_| "test gate".to_owned()), Some(RELEASE)).unwrap();
            queue.Signal(&tail, 1).unwrap();
            let held_behind = !until(RELEASE + Duration::from_millis(300), || reached(&tail));
            ahead.Signal(1).unwrap();
            let held_at_gate = !until(RELEASE / 2, || reached(&tail));
            let opened = until(Duration::from_secs(5), || reached(&tail));
            assert!(opened, "the CPU did not open a gate whose value never came");
            assert!(held_behind, "a gate the queue had not reached was opened");
            assert!(held_at_gate, "the release time ran before the queue reached the gate");
        }
    }

    #[test]
    fn gate_without_release_stays_closed_until_its_value() {
        unsafe {
            let Some((dev, queue)) = device_and_queue() else { return };
            let tail = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let reached = || fence::completed_value(&tail, &dev).is_ok_and(|v| v >= 1);
            let value = Arc::new(AtomicBool::new(false));
            let seen = value.clone();
            arm_with(queue.as_raw(), move || seen.load(Ordering::Acquire), Box::new(|_| "test gate".to_owned()), None).unwrap();
            queue.Signal(&tail, 1).unwrap();
            let held = !until(Duration::from_millis(500), reached);
            value.store(true, Ordering::Release);
            let opened = until(Duration::from_secs(5), reached);
            assert!(opened, "the gate did not open once its value came");
            assert!(held, "a gate without a release time opened without its value");
        }
    }

    #[test]
    fn gate_armed_before_the_last_release_on_its_queue_gets_the_grace() {
        let full = Duration::from_millis(5000);
        let armed = Instant::now();
        let later = armed + Duration::from_millis(10);
        assert_eq!(release_time(full, armed, None), full, "the first gate of a queue gets the full release time");
        assert_eq!(release_time(full, armed, Some(later)), GRACE, "a gate queued behind a released one gets the grace");
        assert_eq!(release_time(full, later, Some(armed)), full, "a gate armed after the release gets the full time");
        let short = Duration::from_millis(100);
        assert_eq!(release_time(short, armed, Some(later)), short, "the grace never lengthens a shorter release time");
    }

    #[test]
    fn gates_queued_behind_a_gate_opened_on_the_cpu_open_after_the_grace() {
        const RELEASE: Duration = Duration::from_millis(1500);
        unsafe {
            let Some((dev, queue)) = device_and_queue() else { return };
            let tail = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let reached = || fence::completed_value(&tail, &dev).is_ok_and(|v| v >= 1);
            for _ in 0..3 {
                arm_with(queue.as_raw(), || false, Box::new(|_| "test gate".to_owned()), Some(RELEASE)).unwrap();
            }
            queue.Signal(&tail, 1).unwrap();
            let start = Instant::now();
            let held = !until(RELEASE - Duration::from_millis(300), reached);
            // one release time and two graces (2 s), not three release times (4.5 s)
            let opened = until(Duration::from_millis(3200).saturating_sub(start.elapsed()), reached);
            assert!(held, "the first gate opened before its release time");
            assert!(opened, "gates armed before the first release each waited a full release time");
        }
    }

    #[test]
    fn gate_marks_its_queue_closed_until_the_cpu_opens_it() {
        unsafe {
            let Some((dev, queue)) = device_and_queue() else { return };
            let tail = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let q = queue.as_raw() as usize;
            let value = Arc::new(AtomicBool::new(false));
            let seen = value.clone();
            arm_with(queue.as_raw(), move || seen.load(Ordering::Acquire), Box::new(|_| "test gate".to_owned()), None).unwrap();
            queue.Signal(&tail, 1).unwrap();
            let closed = closed_on(q);
            value.store(true, Ordering::Release);
            let opened = until(Duration::from_secs(5), || fence::completed_value(&tail, &dev).is_ok_and(|v| v >= 1));
            let reopened = until(Duration::from_secs(1), || !closed_on(q));
            assert!(closed, "an armed gate the CPU has not opened must mark its queue");
            assert!(opened, "the gate did not open once its value came");
            assert!(reopened, "an opened gate still marks its queue");
        }
    }
}
