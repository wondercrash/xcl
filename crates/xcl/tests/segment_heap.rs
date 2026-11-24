//! xcl.exe asks Windows for the segment heap (`xcl.exe.manifest`, embedded by build.rs): DXC, the
//! GPU driver's compiler and the title allocate from the process heap on many threads at once.

#[test]
fn xcl_exe_embeds_the_segment_heap_manifest() {
    let image = std::fs::read(env!("CARGO_BIN_EXE_xcl")).unwrap();
    let text = String::from_utf8_lossy(&image);
    let Some(at) = text.find("SegmentHeap") else { panic!("xcl.exe has no SegmentHeap manifest entry") };
    let element = &text[text[..at].rfind('<').unwrap()..];
    assert!(element.contains("heapType"), "SegmentHeap outside a heapType element: {}", &element[..element.len().min(200)]);
    assert!(text.contains("http://schemas.microsoft.com/SMI/2020/WindowsSettings"), "heapType without its 2020 WindowsSettings namespace");
}
