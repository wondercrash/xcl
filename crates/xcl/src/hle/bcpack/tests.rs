//! Stream tests: truncated input and the modes that have no decoder.

use super::{can_decode, decode, Error};

#[test]
fn truncated_and_empty_streams_fail() {
    assert!(decode(1, &[0xA5; 64], 16 * 8).is_err());
    assert!(decode(1, &[], 16 * 8).is_err());
}

#[test]
fn only_published_modes_decode() {
    assert!(!can_decode(0) && !can_decode(2) && !can_decode(8));
    for mode in [1, 3, 4, 5, 6, 7] {
        assert!(can_decode(mode));
    }
    assert_eq!(decode(2, &[0; 16], 16), Err(Error::Unsupported));
    assert_eq!(decode(1, &[0; 16], 7), Err(Error::OutputSize));
}

#[test]
fn junk_streams_do_not_panic() {
    let mut x = 0x1234_5678u32;
    let junk: Vec<u8> = (0..4096)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        })
        .collect();
    for mode in [1, 3, 4, 5, 6, 7] {
        let block = if mode == 1 || mode == 4 { 8 } else { 16 };
        let _ = decode(mode, &junk, block * 64);
    }
}

/// The BC7 dictionary indices follow VIctory PR #658: a set with 0 bits holds no
/// indices, and pixel 0 is the anchor (top bit clear).
#[test]
fn bc7_dictionary_indices_line_up() {
    for (k, (_, ix)) in super::tables::BC7.iter().enumerate() {
        let hi = 1i16 << (ix.bits[0] - 1);
        let ok = (ix.bits[1] != 0 || ix.ix[1] == [0; 16]) && (0..hi).contains(&ix.ix[0][0]);
        assert!(ok, "entry {k}: {:?}", ix.ix);
    }
}
