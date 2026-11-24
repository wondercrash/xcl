//! `XCL_GPU_BREADCRUMBS=1`: which draw or dispatch a queue that stopped is in.
//!
//! After each draw, dispatch and ExecuteIndirect of console code the PC list writes the op's id
//! twice into a readback buffer (`WriteBufferImmediate`): when the GPU has started it
//! (MARKER_IN) and when it has finished it (MARKER_OUT). Ops console code skipped are noted
//! too, without a GPU write. A reporter thread looks every second for ops started and not
//! finished. One still running after 2 s is logged with its arguments, its shaders (the
//! `XCL_DUMP_SHADERS` dump names), its queue, and the ops before it on its list.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// ring of ops; two words (started, finished) each in the readback buffer
const SLOTS: usize = 1 << 18;
const PC_WRITE_BUFFER_IMMEDIATE: usize = 66;
/// a report names at most this many ops before the stuck one on its list
const BEFORE: usize = 8;

pub(super) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_GPU_BREADCRUMBS").is_ok_and(|v| v.trim() == "1"));
    *ON
}

struct Entry {
    id: u32,
    list: usize,
    pso: usize,
    what: String,
    /// written by the GPU (false: skipped by console code)
    gpu: bool,
}

struct Crumbs {
    _buf: ID3D12Resource,
    gpu_va: u64,
    words: usize,
    next: u32,
    entries: Vec<Option<Entry>>,
    /// PC list → the queue it last went to
    queues: HashMap<usize, usize>,
    /// id → reports that saw it running
    seen: HashMap<u32, u32>,
}

// SAFETY: the readback mapping lives as long as `_buf`; D3D12 resources are free-threaded.
unsafe impl Send for Crumbs {}

static STATE: Mutex<Option<Crumbs>> = Mutex::new(None);

unsafe fn init(list: *mut c_void) -> Option<Crumbs> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
    let dev = dev?;
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
    crate::xlog_info!("D3D12.X: XCL_GPU_BREADCRUMBS: each draw and dispatch of console code marks its start and end; one running for 2 s is logged");
    Some(Crumbs { gpu_va: buf.GetGPUVirtualAddress(), _buf: buf, words: p as usize, next: 0, entries: (0..SLOTS).map(|_| None).collect(), queues: HashMap::new(), seen: HashMap::new() })
}

impl Crumbs {
    fn word(&self, slot: usize, k: usize) -> u32 {
        unsafe { (*(self.words as *const AtomicU32).add(2 * slot + k)).load(Ordering::Acquire) }
    }

    fn state(&self, e: &Entry) -> &'static str {
        let slot = e.id as usize % SLOTS;
        match (e.gpu, self.word(slot, 0) == e.id, self.word(slot, 1) == e.id) {
            (false, ..) => "skipped",
            (true, _, true) => "done",
            (true, true, false) => "running",
            (true, false, false) => "not started",
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

unsafe fn record(list: *mut c_void, pso: usize, what: impl FnOnce() -> String, gpu: bool) {
    if !on() || list.is_null() {
        return;
    }
    let real_pso = super::pso_async::resolve(pso);
    let stand_in = if super::pso_async::is_pending(pso) { " [stand-in: pipeline still being built]" } else { "" };
    push(list, real_pso, format!("{}{stand_in}", what()), gpu);
}

/// Note an op on `list` and, for one that went to the GPU, record its two marker writes there.
unsafe fn push(list: *mut c_void, pso: usize, what: String, gpu: bool) -> Option<u32> {
    let (id, va) = {
        let mut g = STATE.lock().unwrap();
        if g.is_none() {
            *g = init(list);
        }
        let s = g.as_mut()?;
        s.next = s.next.wrapping_add(1).max(1);
        let id = s.next;
        let slot = id as usize % SLOTS;
        for k in 0..2 {
            (*(s.words as *const AtomicU32).add(2 * slot + k)).store(0, Ordering::Release);
        }
        s.entries[slot] = Some(Entry { id, list: list as usize, pso, what, gpu });
        (id, s.gpu_va + 8 * slot as u64)
    };
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
        std::mem::transmute(real::orig_fn(list, PC_WRITE_BUFFER_IMMEDIATE))
    } else {
        std::mem::transmute(*(*(list as *const *const usize)).add(PC_WRITE_BUFFER_IMMEDIATE))
    };
    f(list, 2, params.as_ptr(), modes.as_ptr());
    Some(id)
}

/// `lists` (PC lists) went to `queue`.
pub(super) fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    if !on() {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        for &l in lists {
            s.queues.insert(l as usize, queue as usize);
        }
    }
}

fn shaders(pso: usize) -> String {
    match real::pso_shaders(pso) {
        Some((cs, 0)) => format!("shader {cs:016x}"),
        Some((vs, ps)) => format!("VS {vs:016x} PS {ps:016x}"),
        None => format!("pipeline {pso:#x} (no console shader known)"),
    }
}

/// Log the ops that have been running for 2 s (three reports in a row).
fn report() {
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    let lines: Vec<String> = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        let mut seen = HashMap::new();
        let mut stuck = Vec::new();
        for e in s.entries.iter().flatten() {
            if e.gpu && s.state(e) == "running" {
                let n = s.seen.get(&e.id).copied().unwrap_or(0) + 1;
                seen.insert(e.id, n);
                if n == 3 {
                    stuck.push(e.id);
                }
            }
        }
        s.seen = seen;
        let s = &*s;
        stuck.iter().take(16).map(|&id| {
            let e = s.entries[id as usize % SLOTS].as_ref().unwrap();
            let queue = s.queues.get(&e.list).map_or_else(|| "a queue not seen".to_owned(), |&q| super::list_x::queue_label(q));
            let mut before: Vec<&Entry> = s.entries.iter().flatten().filter(|o| o.list == e.list && o.id < e.id).collect();
            before.sort_unstable_by_key(|o| o.id);
            let skipped = before.iter().filter(|o| !o.gpu).count();
            let before: Vec<String> = before.iter().rev().take(BEFORE).rev().map(|o| format!("#{} {} ({}, {})", o.id, o.what, shaders(o.pso), s.state(o))).collect();
            let after = s.entries.iter().flatten().filter(|o| o.list == e.list && o.id > e.id).count();
            format!(
                "D3D12.X: breadcrumb #{} on {queue} has been running for 2 s: {} ({}) on list {:#x}; {after} later op(s) on the list, {skipped} skipped before it; before it: [{}]",
                e.id, e.what, shaders(e.pso), e.list, before.join("; ")
            )
        }).collect()
    };
    for l in lines {
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 256 {
            crate::xlog_warn!("{l}");
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
        let e = s.entries[id as usize % SLOTS].as_ref().filter(|e| e.id == id).expect("entry overwritten");
        s.state(e)
    }

    #[test]
    fn breadcrumbs_mark_start_and_end_of_the_ops_a_list_ran() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let ran = push(list.as_raw(), 0, "first".into(), true).expect("no breadcrumb buffer");
            let skip = push(list.as_raw(), 0, "skipped".into(), false).unwrap();
            list.Close().unwrap();
            assert_eq!(state_of(ran), "not started");
            assert_eq!(state_of(skip), "skipped");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            executed(queue.as_raw(), &[list.as_raw()]);
            queue.Signal(&fence, 1).unwrap();
            let start = Instant::now();
            while fence.GetCompletedValue() < 1 && start.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(fence.GetCompletedValue(), 1, "queue did not finish");
            assert_eq!(state_of(ran), "done");
        }
    }
}
