use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};
use xcl::hle::kernelx::{gpu_view_map_gen, in_graphics_reservation, mirrors, VirtualAlloc, VirtualFree};

// Only these tests run in this process and change the generation.
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn reusing_a_gpu_view_with_new_backing_notifies_its_textures() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Bookkeeping only. Supply the same GPU address on both reservations.
    let (first, second, gpu, size) = (0x10_0000, 0x20_0000, 0x30_0000, 0x2_0000);
    let reserve = |base, p| mirrors::reserved(base, size, MEM_RESERVE | 0x2040_0000, p);
    reserve(first, PAGE_READWRITE);
    reserve(gpu, PAGE_NOACCESS | 0x800);
    assert_eq!(mirrors::source(gpu + 65536), Some(first + 65536));
    mirrors::released(gpu);
    mirrors::released(first);
    let unpaired = gpu_view_map_gen();
    assert_eq!(mirrors::source(gpu + 65536), None);
    reserve(second, PAGE_READWRITE);
    assert_eq!(gpu_view_map_gen(), unpaired);
    reserve(gpu, PAGE_NOACCESS | 0x800);
    assert_eq!(mirrors::source(gpu + 65536), Some(second + 65536));
    assert_eq!(gpu_view_map_gen(), unpaired + 1, "notify a surviving texture of the new backing");
    mirrors::released(second);
    mirrors::released(gpu);
}

#[test]
fn sparse_64k_views_pair_and_forget_released_addresses() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(uma) = xcl::gpu::uma::get().filter(|u| u.has_sparse()) else {
        xcl::gpu::recomp::runner::skip_test("no sparse UMA address space");
        return;
    };
    let size = 3 * 65536;
    let reserve = |size, protection| {
        let p = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | 0x2040_0000, protection) };
        assert!(!p.is_null(), "failed to reserve {size:#x} bytes with protection {protection:#x}");
        p
    };
    let gpu_protection = PAGE_NOACCESS | 0x800; // PAGE_GRAPHICS_NOACCESS
    let before = gpu_view_map_gen();
    let cpu = reserve(size, PAGE_READWRITE);
    assert_eq!(gpu_view_map_gen(), before);
    let gpu = reserve(size, gpu_protection);
    assert!(uma.is_sparse(cpu as usize) && uma.is_sparse(gpu as usize));
    let sample = gpu as usize + 65536;
    let backing = cpu as usize + 65536;
    assert_eq!(mirrors::source(sample), Some(backing));

    unsafe {
        let paired = gpu_view_map_gen();
        assert_eq!(paired, before + 1, "notify when the sparse views first pair");
        assert_eq!(VirtualFree(gpu, size, MEM_RELEASE), 0);
        assert_eq!(VirtualFree(gpu, size, MEM_DECOMMIT), 1);
        assert_eq!(gpu_view_map_gen(), paired);
        assert_eq!(mirrors::source(sample), Some(backing));

        assert_eq!(VirtualFree(gpu, 0, MEM_RELEASE), 1);
        assert_eq!(mirrors::source(sample), None);
        assert_eq!(gpu_view_map_gen(), paired + 1);
        let replacement = reserve(size, PAGE_READWRITE);
        assert_eq!(mirrors::source(replacement as usize + 65536), None);
        assert_eq!(VirtualFree(replacement, 0, MEM_RELEASE), 1);
        assert_eq!(gpu_view_map_gen(), paired + 1);

        let gpu_again = reserve(size, gpu_protection);
        let sample = gpu_again as usize + 65536;
        assert_eq!(mirrors::source(sample), Some(backing));
        assert_eq!(gpu_view_map_gen(), paired + 2);
        let paired = gpu_view_map_gen();
        assert_eq!(VirtualFree(cpu, 0, MEM_RELEASE), 1);
        assert_eq!(mirrors::source(sample), None);
        assert_eq!(gpu_view_map_gen(), paired + 1);
        assert_eq!(VirtualFree(gpu_again, 0, MEM_RELEASE), 1);
        assert_eq!(gpu_view_map_gen(), paired + 1);

        // Two reservations this large exceed all sparse windows together. These flags contain
        // neither MEM_GRAPHICS nor MEM_COMMIT, so there is no coherent-window fallback.
        let sparse_bytes: usize = uma.watch_regions().into_iter()
            .filter(|&(base, _)| uma.is_sparse(base)).map(|(_, len)| len).sum();
        let size = sparse_bytes / 2 + 65536;
        let cpu = reserve(size, PAGE_READWRITE);
        let gpu = reserve(size, gpu_protection);
        assert!(uma.is_sparse(cpu as usize));
        assert!(!uma.contains(gpu as usize) && in_graphics_reservation(gpu as usize));
        assert_eq!(mirrors::source(gpu as usize + 65536), Some(cpu as usize + 65536));
        let paired = gpu_view_map_gen();
        assert_eq!(VirtualFree(gpu, 0, MEM_RELEASE), 1);
        assert_eq!(mirrors::source(gpu as usize + 65536), None);
        assert_eq!(gpu_view_map_gen(), paired + 1);
        assert_eq!(VirtualFree(cpu, 0, MEM_RELEASE), 1);
    }
}
