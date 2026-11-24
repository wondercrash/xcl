//! `XCL_TITLE_WATCHDOG=0` (bring-up only, temporary): a retail title's own hang watchdog no longer
//! ends the process. Its watchdog thread wakes every second; when the heartbeat its frame loop
//! stores (a QueryPerformanceCounter value) is unchanged for 30 checks in a row – or, while a
//! thread holds a "long operation" scope, when one is older than 300 s and a tunable enables
//! that check – it calls a registered callback that writes to address 0 (0xC0000005 in the
//! title's code). A driver's first compile of a big pipeline holds the title's thread longer than
//! that, so a run with cold caches ends there. Three branches of that one function become no-ops:
//! the two 30-check countdowns (one per variant of its loop) stop counting and the 300 s compare
//! never takes its jump; the watchdog thread itself keeps running and still exits on the title's
//! shutdown flag. Only for the one executable the sites were read from (PE timestamp, image size
//! and the code around each site); anything else is left as it is. With it on, a real hang no
//! longer ends the run either: the fix is removing the stalls, not this.

/// A branch to defuse: `expect` is the code at `rva`, `patch` replaces `expect[at..]`.
struct Site {
    rva: u32,
    expect: &'static [u8],
    at: usize,
    patch: &'static [u8],
}

const TIMESTAMP: u32 = 0x6AF7_83D4;
const SIZE_OF_IMAGE: u32 = 0x0B06_D000;

const SITES: [Site; 3] = [
    // a thread's open scope older than 300 s: `vucomiss xmm8, xmm7; jae crash` → `jae` is a nop
    Site {
        rva: 0x0221_5BC2,
        expect: &[
            0xC5, 0x7A, 0x59, 0x05, 0xF6, 0x99, 0xC3, 0x07, 0xB9, 0xD6, 0xE1, 0xAE, 0x89, 0xE8, 0x9C, 0xE2, 0x7B, 0xFE, 0x84, 0xC0, 0x74, 0x06,
            0xC5, 0x78, 0x2E, 0xC7, 0x73, 0x2E, 0x48, 0xFF, 0xC7,
        ],
        at: 0x1A,
        patch: &[0x66, 0x90],
    },
    // heartbeat unchanged (first loop): `dec ebp; je crash` → nop, the count stays at 30
    Site {
        rva: 0x0221_5C5D,
        expect: &[
            0x48, 0x8B, 0x05, 0xD4, 0x06, 0xFA, 0x04, 0xBB, 0x1E, 0x00, 0x00, 0x00, 0x49, 0x39, 0xC7, 0x75, 0xD6, 0xFF, 0xCD, 0x74, 0x04, 0x89,
            0xEB, 0xEB, 0xCB, 0x31, 0xDB, 0xEB, 0x9C,
        ],
        at: 0x11,
        patch: &[0x0F, 0x1F, 0x40, 0x00],
    },
    // heartbeat unchanged (second loop): the same
    Site {
        rva: 0x0221_5D71,
        expect: &[
            0x48, 0x8B, 0x0D, 0xC0, 0x05, 0xFA, 0x04, 0xB8, 0x1E, 0x00, 0x00, 0x00, 0x49, 0x39, 0xCF, 0x75, 0x38, 0xFF, 0xCD, 0x74, 0x04, 0x89,
            0xE8, 0xEB, 0x2D,
        ],
        at: 0x11,
        patch: &[0x0F, 0x1F, 0x40, 0x00],
    },
];

pub fn requested() -> bool {
    std::env::var("XCL_TITLE_WATCHDOG").is_ok_and(|v| v.trim() == "0")
}

/// Whether an image is the executable the sites were read from (`bytes_at(rva, len)`: its code).
fn identify<'a>(timestamp: u32, size_of_image: u32, bytes_at: impl Fn(u32, usize) -> Option<&'a [u8]>) -> Result<(), String> {
    if timestamp != TIMESTAMP || size_of_image != SIZE_OF_IMAGE {
        return Err(format!(
            "PE timestamp {timestamp:#x}, image size {size_of_image:#x}; expected {TIMESTAMP:#x}, {SIZE_OF_IMAGE:#x}"
        ));
    }
    match SITES.iter().find(|s| bytes_at(s.rva, s.expect.len()) != Some(s.expect)) {
        Some(s) => Err(format!("other code at +{:#x}", s.rva)),
        None => Ok(()),
    }
}

/// Defuses the watchdog in `image` (the mapped image from offset 0) once every site matched.
fn defuse(timestamp: u32, size_of_image: u32, image: &mut [u8]) -> Result<(), String> {
    identify(timestamp, size_of_image, |rva, len| image.get(rva as usize..rva as usize + len))?;
    for s in &SITES {
        let at = s.rva as usize + s.at;
        image[at..at + s.patch.len()].copy_from_slice(s.patch);
    }
    Ok(())
}

/// The main image, mapped at `base` (`size` bytes) and still writable: the loader calls this
/// before it sets the sections' page protections, so no code of the image has run.
pub unsafe fn apply(timestamp: u32, base: usize, size: usize, name: &str) {
    if !requested() {
        return;
    }
    let image = std::slice::from_raw_parts_mut(base as *mut u8, size);
    match defuse(timestamp, size as u32, image) {
        Ok(()) => crate::xlog_info!(
            "XCL_TITLE_WATCHDOG=0: {name}'s hang watchdog is off (it writes to address 0 after 30 s without a frame) – a real hang will no longer end the run either"
        ),
        Err(why) => crate::xlog_warn!("XCL_TITLE_WATCHDOG=0: {name} is not the executable whose hang watchdog XCL knows ({why}) – left as it is"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image of the expected size up to the last site, the expected code at every site.
    fn image() -> Vec<u8> {
        let end = SITES.iter().map(|s| s.rva as usize + s.expect.len()).max().unwrap();
        let mut v = vec![0u8; end + 0x100];
        for s in &SITES {
            v[s.rva as usize..s.rva as usize + s.expect.len()].copy_from_slice(s.expect);
        }
        v
    }

    #[test]
    fn defuses_the_known_executable_once() {
        let mut v = image();
        defuse(TIMESTAMP, SIZE_OF_IMAGE, &mut v).unwrap();
        for s in &SITES {
            let mut want = s.expect.to_vec();
            want[s.at..s.at + s.patch.len()].copy_from_slice(s.patch);
            assert_eq!(&v[s.rva as usize..s.rva as usize + want.len()], &want[..], "site +{:#x}", s.rva);
        }
        // patched code is not the expected code any more: a second call changes nothing
        let before = v.clone();
        assert!(defuse(TIMESTAMP, SIZE_OF_IMAGE, &mut v).is_err());
        assert_eq!(v, before);
    }

    #[test]
    fn leaves_other_executables_alone() {
        let untouched = image();
        for (ts, size) in [(TIMESTAMP + 1, SIZE_OF_IMAGE), (TIMESTAMP, SIZE_OF_IMAGE + 0x1000)] {
            let mut v = untouched.clone();
            assert!(defuse(ts, size, &mut v).unwrap_err().contains("PE timestamp"));
            assert_eq!(v, untouched);
        }
        // one byte of the last site differs: no site is patched
        let mut v = untouched.clone();
        let last = &SITES[2];
        v[last.rva as usize] ^= 0xFF;
        let changed = v.clone();
        assert_eq!(defuse(TIMESTAMP, SIZE_OF_IMAGE, &mut v).unwrap_err(), format!("other code at +{:#x}", last.rva));
        assert_eq!(v, changed);
        // an image too short for a site
        let mut short = untouched[..SITES[0].rva as usize + 4].to_vec();
        assert!(defuse(TIMESTAMP, SIZE_OF_IMAGE, &mut short).is_err());
    }

    #[test]
    fn patches_stay_inside_their_sites() {
        for s in &SITES {
            assert!(s.at + s.patch.len() <= s.expect.len(), "site +{:#x}", s.rva);
        }
    }

    /// The real executable (`XCL_SAMPLE_EXE` = its file) is the one the sites were read from.
    #[test]
    fn sample_exe_is_the_known_executable() {
        let Ok(path) = std::env::var("XCL_SAMPLE_EXE") else {
            println!("XCL_SAMPLE_EXE not set – skipping");
            return;
        };
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read XCL_SAMPLE_EXE '{path}': {e}"));
        let pe = crate::pe::Pe::parse(&data, crate::pe::Layout::File).unwrap();
        identify(pe.timestamp, pe.size_of_image, |rva, len| pe.bytes_at(rva, len)).unwrap();
    }
}
