//! Queue-side gates for recorded waits on CPU-visible progress values.

use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute, ManuallyDrop};
use std::sync::{mpsc, LazyLock};
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

static TX: LazyLock<Result<mpsc::Sender<Gate>, String>> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new().name("xcl-wait-gates".into()).spawn(move || monitor(rx))
        .map_err(|e| format!("starting the Wait64BitValueX monitor: {e}"))?;
    Ok(tx)
});

struct Gate {
    queue: usize,
    device: ID3D12Device,
    // D3D12 can still be using this fence until the queue passes the retirement signal.
    // Only release it on confirmed retirement/removal, even if the worker exits unexpectedly.
    fence: ManuallyDrop<ID3D12Fence>,
    ready: Option<Box<dyn Fn() -> bool + Send>>,
    warn_at: Option<Instant>,
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

    fn poll(&mut self) -> Poll {
        let value = match unsafe { fence::completed_value(&self.fence, &self.device) } {
            Ok(value) => value,
            Err(e) => return self.failed(e),
        };
        if self.retire_at.is_some_and(|target| value >= target) {
            return Poll::Retired;
        }
        let Some(ready) = &self.ready else { return Poll::Pending };
        if value < REACHED || !ready() {
            if self.warn_at.is_some_and(|at| Instant::now() >= at) {
                self.warn_at = None;
                let waiting_on = if value < REACHED { "preceding queue work" } else { "the memory condition" };
                crate::xlog_warn!("D3D12.X: Wait64BitValueX gate on queue {:#x} still waiting on {waiting_on} after {} s; leaving the queue blocked", self.queue, WARN_AFTER.as_secs());
            }
            return Poll::Pending;
        }
        if let Err(e) = unsafe { self.fence.Signal(OPEN) } {
            return self.failed(unsafe { fence::api_error(&self.device, "ID3D12Fence::Signal", e) });
        }
        self.ready = None;
        if self.retire_at.is_some() { Poll::Pending } else { Poll::KeepFence }
    }
}

fn monitor(rx: mpsc::Receiver<Gate>) {
    let mut pending = Vec::new();
    loop {
        if pending.is_empty() {
            let Ok(gate) = rx.recv() else { return };
            pending.push(gate);
        }
        pending.extend(rx.try_iter());
        let mut i = 0;
        while i < pending.len() {
            match pending[i].poll() {
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
/// opens it only when `ready` observes the requested value; the submitter is free to submit the
/// producer on another queue.
///
/// `queue` must be a live native D3D12 command queue. `ready` must be safe to call on the monitor.
pub(super) unsafe fn arm(queue: *mut c_void, ready: impl Fn() -> bool + Send + 'static) -> Result<(), String> {
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
        device,
        fence: ManuallyDrop::new(fence),
        ready: armed.then(|| Box::new(ready) as Box<dyn Fn() -> bool + Send>),
        warn_at: Some(Instant::now() + WARN_AFTER),
        retire_at,
    }).map_err(|_| "Wait64BitValueX monitor stopped after submitting to the queue; cannot service the gate".to_owned())?;
    wait
}
