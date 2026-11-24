use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn disasm_honors_byte_range_options() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let shader =
        std::env::temp_dir().join(format!("xcl-disasm-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&shader, [0, 0, 0x80, 0xbf, 0, 0, 0x81, 0xbf]).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("disasm")
            .arg(&shader)
            .args(args)
            .output()
    };
    let length = run(&["--length", "4"]);
    let offset = run(&["--offset", "0x4", "--length", "0x4"]);
    let malformed = run(&["--offset", "bogus"]);
    let missing = run(&["--length"]);
    std::fs::remove_file(&shader).unwrap();

    let length = length.unwrap();
    assert!(length.status.success());
    let listing = String::from_utf8(length.stdout).unwrap();
    assert!(listing.contains("00000000: s_nop"), "{listing}");
    assert!(!listing.contains("s_endpgm"), "{listing}");

    let offset = offset.unwrap();
    assert!(offset.status.success());
    let listing = String::from_utf8(offset.stdout).unwrap();
    assert!(listing.contains("00000004: s_endpgm"), "{listing}");
    assert!(!listing.contains("s_nop"), "{listing}");

    for (output, flag) in [(malformed, "--offset"), (missing, "--length")] {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8(output.stderr).unwrap().contains(flag));
    }
}

#[test]
fn disasm_prints_vop3_literal_and_advances_pc() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let shader =
        std::env::temp_dir().join(format!("xcl-disasm-literal-{}-{nonce}.bin", std::process::id()));
    // V_BFE_U32 with a literal in SRC2, followed by S_ENDPGM.
    // https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/llvm/test/MC/AMDGPU/vop3-literal.s#L22-L23
    std::fs::write(
        &shader,
        [0x00, 0x00, 0x48, 0xd5, 0x01, 0x03, 0xfc, 0x03, 0x39, 0x30, 0x00, 0x00, 0x00, 0x00, 0x81, 0xbf],
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(&shader)
        .args(["--offset", "0"])
        .output();
    std::fs::remove_file(&shader).unwrap();

    let output = output.unwrap();
    assert!(output.status.success(), "{:?}", output);
    let listing = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        listing.lines().skip(1).collect::<Vec<_>>(),
        [
            format!(
                "  00000000: {:<52} 00 00 48 d5 01 03 fc 03 39 30 00 00",
                "v_bfe_u32 v0, v1, s1, 0x3039"
            ),
            format!("  0000000c: {:<52} 00 00 81 bf", "s_endpgm"),
        ]
    );
}
