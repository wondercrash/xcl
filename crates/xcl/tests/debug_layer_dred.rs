//! The D3D12 debug layer (`XCL_D3D12_DEBUG=1`) with DRED (`XCL_DRED=1`): making a command list,
//! the runtime makes DRED's breadcrumb buffer through the device's CreatePlacedResource – XCL's
//! hook – and gets back its own inner resource, which XCL adopts. Its QueryInterface answers
//! with the debug layer's object; releasing that with the inner object's Release ended a
//! retail title 2.7 s in (0xC0000005 in D3D12Core, read at 0xffffffffffffffff).

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::real;

#[test]
fn command_list_with_debug_layer_and_dred() {
    std::env::set_var("XCL_UMA_MB", "64");
    std::env::set_var("XCL_D3D12_DEBUG", "1");
    std::env::set_var("XCL_DRED", "1");
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        return runner::skip_test("no GPU");
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.expect("backend device").clone();
    if dev.cast::<ID3D12InfoQueue>().is_err() {
        return runner::skip_test("no D3D12 debug layer (Graphics Tools)");
    }
    unsafe {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("command list");
        list.Close().expect("Close");
        alloc.Reset().expect("allocator Reset");
        list.Reset(&alloc, None).expect("list Reset");
        list.Close().expect("second Close");
    }
}
