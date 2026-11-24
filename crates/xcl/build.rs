use std::path::Path;

// `pso.rs` also contains cache storage and pipeline setup. Keep it out of the source hash so
// those changes do not discard translations; bump this only when its container parsing,
// translation options, profiles, or post-translation HLSL changes.
const TRANSLATION_GLUE_REVISION: &str = "pso-glue-v1";

const TRANSLATOR_SOURCES: &[&str] = &[
    "src/gpu/dxbc.rs",
    "src/gpu/rdna2.rs",
    "src/gpu/rdna2_spec.rs",
    "src/gpu/rdna2_xbox.rs",
    "src/gpu/recomp/dxc.rs",
    "src/gpu/recomp/emit.rs",
    "src/gpu/recomp/mesh_wave.rs",
    "src/gpu/recomp/mod.rs",
    "src/gpu/recomp/upper_lanes.rs",
    "src/gpu/recomp/x64_masks.rs",
    "src/gpu/shader.rs",
    "src/gpu/xbox.rs",
];

/// The per-lane form's sources (`XCL_PER_LANE`): part of only its own cache entries' identity,
/// so editing them does not invalidate the HLSL path's.
const PER_LANE_SOURCES: &[&str] = &[
    "src/gpu/recomp/lane_masks.rs",
    "src/gpu/recomp/lower.rs",
    "src/gpu/recomp/lower/wave.rs",
    "src/gpu/recomp/lower_dxil.rs",
    "src/gpu/recomp/ssa.rs",
];

fn add(h: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

fn main() {
    // xcl.exe without high-entropy ASLR: the title's memory then lies under 2^40, which a
    // retail title's BVH builder requires of its GPU addresses (#585, PR #631)
    match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
        Ok("msvc") => println!("cargo:rustc-link-arg-bin=xcl=/HIGHENTROPYVA:NO"),
        Ok("gnu") => println!("cargo:rustc-link-arg-bin=xcl=-Wl,--disable-high-entropy-va"),
        _ => {}
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // xcl.exe asks for the segment heap (xcl.exe.manifest). Under the NT heap's one lock, 15 DXC
    // threads made the NVIDIA driver's in-process pipeline compiles 2.4 times as slow.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rerun-if-changed=xcl.exe.manifest");
        println!("cargo:rustc-link-arg-bin=xcl=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg-bin=xcl=/MANIFESTINPUT:{}", root.join("xcl.exe.manifest").display());
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    add(&mut h, b"xcl-translator-source-v1\0");
    add(&mut h, TRANSLATION_GLUE_REVISION.as_bytes());
    add(&mut h, &[0]);
    for relative in TRANSLATOR_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut h, relative.as_bytes());
        add(&mut h, &[0]);
        add(&mut h, &std::fs::read(root.join(relative)).expect("read shader translator source"));
        add(&mut h, &[0]);
    }
    println!("cargo:rustc-env=XCL_TRANSLATOR_SOURCE_ID={h:016x}");
    let mut p = 0xcbf2_9ce4_8422_2325u64;
    for relative in PER_LANE_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut p, relative.as_bytes());
        add(&mut p, &[0]);
        add(&mut p, &std::fs::read(root.join(relative)).expect("read per-lane source"));
        add(&mut p, &[0]);
    }
    println!("cargo:rustc-env=XCL_PER_LANE_SOURCE_ID={p:016x}");
}
