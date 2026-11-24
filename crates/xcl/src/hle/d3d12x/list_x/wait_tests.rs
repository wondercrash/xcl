use super::super::{bind, present, real, title_list};
use super::{execute_with_waits_on, on_create, on_queue_create, wait64_x, write64_bottom_x};
use crate::gpu::{fence, recomp::runner::skip_test};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

const TIMEOUT: Duration = Duration::from_secs(5);

fn until(timeout: Duration, mut ready: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while !ready() {
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

unsafe fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut resource).unwrap();
    resource.unwrap()
}

unsafe fn write_upload_words(upload: &ID3D12Resource, words: [u64; 2]) {
    let mut p = std::ptr::null_mut();
    upload.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut p)).unwrap();
    p.cast::<[u64; 2]>().write_unaligned(words);
    upload.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }));
}

unsafe fn readback_words(readback: &ID3D12Resource) -> [u64; 2] {
    let mut p = std::ptr::null_mut();
    readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut p)).unwrap();
    let words = p.cast::<[u64; 2]>().read_unaligned();
    readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
    words
}

unsafe fn title_list(dev: &ID3D12Device, alloc: &ID3D12CommandAllocator, ty: D3D12_COMMAND_LIST_TYPE) -> ID3D12GraphicsCommandList {
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, alloc, None).unwrap();
    let pc = list.into_raw();
    real::adopt(pc, real::Kind::CommandList);
    on_create(pc, alloc.as_raw() as usize, 0, ty);
    bind::reset(pc, alloc.as_raw() as usize, 0);
    let wrapped = ID3D12GraphicsCommandList::from_raw(title_list::wrap(pc));
    assert_ne!(wrapped.as_raw(), pc, "expected the default title list wrapper");
    wrapped
}

#[test]
fn cross_queue_wait64_runs_at_its_recorded_position() {
    const DIRECT: D3D12_COMMAND_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_DIRECT;
    const COMPUTE: D3D12_COMMAND_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    const GE: u32 = D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32;
    const READY_VALUE: u64 = 0x1_0000_0041;
    const DONE_VALUE: u64 = 0x2_0000_0042;
    const STALE: u64 = 0x0102_0304_0506_0708;
    const UPDATED: u64 = 0xA1B2_C3D4_E5F6_0798;

    unsafe {
        let mut dev = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            skip_test("no GPU");
            return;
        }
        let dev: ID3D12Device = dev.unwrap();
        let queue = |ty| -> ID3D12CommandQueue {
            dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap()
        };
        let (a, b) = (queue(DIRECT), queue(COMPUTE));
        let new_fence = || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let (a_tail, b_tail, release_b) = (new_fence(), new_fence(), new_fence());
        let reached = |f, at| fence::completed_value(f, &dev).is_ok_and(|v| v >= at);
        let both_tails = |at| reached(&a_tail, at) && reached(&b_tail, at);

        let upload = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let value = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        let readback = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        write_upload_words(&upload, [STALE, UPDATED]);

        let init_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(DIRECT).unwrap();
        let init: ID3D12GraphicsCommandList = dev.CreateCommandList(0, DIRECT, &init_alloc, None).unwrap();
        init.CopyBufferRegion(&value, 0, &upload, 0, 8);
        init.CopyBufferRegion(&readback, 0, &upload, 0, 8);
        init.CopyBufferRegion(&readback, 8, &upload, 0, 8);
        init.Close().unwrap();
        a.ExecuteCommandLists(&[Some(init.cast().unwrap())]);
        a.Signal(&a_tail, 1).unwrap();
        assert!(until(TIMEOUT, || reached(&a_tail, 1)), "initial GPU copies did not finish");

        for (queue, ty) in [(&a, DIRECT), (&b, COMPUTE)] {
            real::adopt(queue.as_raw(), real::Kind::CommandQueue);
            on_queue_create(queue.as_raw() as usize, dev.as_raw() as usize, ty);
        }
        let a_alloc = dev.CreateCommandAllocator(DIRECT).unwrap();
        let b_alloc = dev.CreateCommandAllocator(COMPUTE).unwrap();
        let (a_list, b_list) = (title_list(&dev, &a_alloc, DIRECT), title_list(&dev, &b_alloc, COMPUTE));
        // Use guest-style committed pages, not PE image sections that Wine can map
        // as write-copy. Keep the allocation alive if a failed run leaves work pending.
        let cells = VirtualAlloc(std::ptr::null(), 16, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<[AtomicU64; 2]>();
        assert!(!cells.is_null());
        let [ready_cell, done_cell] = &*cells;
        let rescue = || {
            ready_cell.store(READY_VALUE, Ordering::Release);
            done_cell.store(DONE_VALUE, Ordering::Release);
        };
        let ready = ready_cell.as_ptr() as u64;
        let done = done_cell.as_ptr() as u64;

        // A GPU canary before the wait makes a recording-time prefix submit visible even if
        // the deferred ready write has not yet landed on the CPU.
        a_list.CopyBufferRegion(&readback, 8, &upload, 8, 8);
        write64_bottom_x(title_list::active(a_list.as_raw()), ready, READY_VALUE, 0, 0);
        wait64_x(title_list::active(a_list.as_raw()), done, GE, DONE_VALUE, 0, 0);
        a_list.CopyBufferRegion(&readback, 0, &value, 0, 8);
        a_list.Close().unwrap();

        wait64_x(title_list::active(b_list.as_raw()), ready, GE, READY_VALUE, 0, 0);
        b_list.CopyBufferRegion(&value, 0, &upload, 8, 8);
        write64_bottom_x(title_list::active(b_list.as_raw()), done, DONE_VALUE, 0, 0);
        b_list.Close().unwrap();

        a.Signal(&a_tail, 2).unwrap();
        b.Signal(&b_tail, 2).unwrap();
        let recording_idle = until(TIMEOUT, || both_tails(2));
        let recording_untouched = recording_idle && readback_words(&readback) == [STALE, STALE]
            && ready_cell.load(Ordering::Acquire) == 0 && done_cell.load(Ordering::Acquire) == 0;
        if !recording_untouched {
            rescue();
            let _ = until(TIMEOUT, || both_tails(2));
            present::forget_test_queue(a.as_raw());
            present::forget_test_queue(b.as_raw());
        }
        assert!(recording_untouched, "recording or closing a list submitted GPU work: queue tails reached = {recording_idle}");

        // Hold B's GPU work so A cannot legitimately finish early. Both XCL submissions
        // happen on ONE thread, in consumer-first order, before releasing the external fence.
        let (staged, held, unblocked, submit) = std::thread::scope(|s| {
            b.Wait(&release_b, 1).unwrap();
            let submit = std::thread::Builder::new().spawn_scoped(s, || {
                // Execute through the adopted queue's production hook, passing the wrappers.
                (a.vtable().ExecuteCommandLists)(a.as_raw(), 1, &a_list.as_raw());
                (b.vtable().ExecuteCommandLists)(b.as_raw(), 1, &b_list.as_raw());
                a.Signal(&a_tail, 3).and_then(|_| b.Signal(&b_tail, 3))
            });
            let staged = submit.as_ref().is_ok_and(|t| {
                until(TIMEOUT, || t.is_finished() && ready_cell.load(Ordering::Acquire) == READY_VALUE)
            });
            let held = staged && !until(Duration::from_millis(100), || reached(&a_tail, 3))
                && done_cell.load(Ordering::Acquire) == 0;
            let unblocked = release_b.Signal(1);
            if !staged || unblocked.is_err() {
                rescue();
            }
            let submit = submit.map(|t| t.join());
            (staged, held, unblocked, submit)
        });

        let finished = until(TIMEOUT, || both_tails(3) && done_cell.load(Ordering::Acquire) == DONE_VALUE);
        if !finished {
            rescue();
            // Queue the drain markers after any work submitted before an error.
            let _ = a.Signal(&a_tail, 4);
            let _ = b.Signal(&b_tail, 4);
            let _ = until(TIMEOUT, || both_tails(4));
        }
        present::forget_test_queue(a.as_raw());
        present::forget_test_queue(b.as_raw());

        unblocked.expect("release external producer fence");
        submit.expect("spawn submit thread").expect("submit thread panicked").expect("queue tail signals");
        assert!(staged, "consumer prefix did not publish ready with both submissions returned");
        let observed = finished.then(|| readback_words(&readback));
        assert_eq!(observed, Some([UPDATED, UPDATED]), "GPU wait chain timed out or copied stale data");
        assert!(held, "consumer or producer finished while producer's external fence was still held");
        assert_ne!(VirtualFree(cells.cast(), 0, MEM_RELEASE), 0);
    }
}

#[test]
fn ordinary_copy_runs_before_wait64_for_late_external_value() {
    const DIRECT: D3D12_COMMAND_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_DIRECT;
    const GE: u32 = D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32;
    const READY: u64 = 0x1_0000_0041;
    const WORDS: [u64; 2] = [0x0102_0304_0506_0708, 0xA1B2_C3D4_E5F6_0798];

    unsafe {
        let mut dev = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            skip_test("no GPU");
            return;
        }
        let dev: ID3D12Device = dev.unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: DIRECT, ..Default::default() }).unwrap();
        let completion: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let reached = |at| fence::completed_value(&completion, &dev).is_ok_and(|v| v >= at);
        let upload = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let prefix = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let suffix = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        write_upload_words(&upload, WORDS);
        let mut prefix_data = std::ptr::null_mut();
        prefix.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut prefix_data)).unwrap();

        let alloc = dev.CreateCommandAllocator(DIRECT).unwrap();
        let list = title_list(&dev, &alloc, DIRECT);
        let ready = VirtualAlloc(std::ptr::null(), 8, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<AtomicU64>();
        assert!(!ready.is_null());
        list.CopyBufferRegion(&prefix, 0, &upload, 0, 16);
        wait64_x(title_list::active(list.as_raw()), ready as u64, GE, READY, 0, 0);
        list.CopyBufferRegion(&suffix, 0, &upload, 0, 16);
        list.Close().unwrap();

        let mut pieces = Vec::new();
        assert!(title_list::push_segments(list.as_raw(), &mut pieces));
        // Use the queue scheduler's native submission path, with a real completion fence
        // after each batch so the CPU never reads a buffer still being written by the GPU.
        let mut batches = 0;
        let mut signal_error = None;
        execute_with_waits_on(queue.as_raw(), &pieces, |chunk| {
            (queue.vtable().ExecuteCommandLists)(queue.as_raw(), chunk.len() as u32, chunk.as_ptr());
            batches += 1;
            if let Err(err) = queue.Signal(&completion, batches) {
                signal_error = Some(err);
            }
        });
        let tail = queue.Signal(&completion, batches + 1);
        let observed_prefix = until(TIMEOUT, || reached(1))
            .then(|| prefix_data.cast::<[u64; 2]>().read_unaligned());
        let suffix_held = observed_prefix.is_some() && !until(Duration::from_millis(100), || reached(2));
        prefix.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        // This is the external producer. Release even on RED so queued work can drain.
        (*ready).store(READY, Ordering::Release);
        let finished = tail.is_ok() && until(TIMEOUT, || reached(batches + 1));
        if finished {
            assert_ne!(VirtualFree(ready.cast(), 0, MEM_RELEASE), 0);
        }
        assert!(signal_error.is_none(), "signal batch completion: {signal_error:?}");
        tail.expect("signal queue tail");
        assert!(finished, "queue did not drain after the producer advanced");
        assert_eq!(observed_prefix, Some(WORDS), "ordinary copy did not finish before the producer advanced");
        assert_eq!(batches, 2, "expected a prefix and a continuation");
        assert!(suffix_held, "suffix completed before the producer advanced");
        assert_eq!(readback_words(&suffix), WORDS);
    }
}
