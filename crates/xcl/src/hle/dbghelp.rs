//! dbghelp: a title's crash handler writes a minidump of its own process (`MiniDumpWriteDump`
//! from its dump thread, while the crashed thread waits for it). In-process, dbgcore suspends
//! every other thread and then takes the process heap's lock – held, often enough, by one of the
//! threads it has just suspended: the dump never finishes, the crashed thread never returns and
//! the process hangs with XCL's window frozen (a retail title's world-loading crash: it ended
//! within a second once, and hung on the next run). XCL has logged the exception and every
//! thread by then, so a dump of the process itself is declined; dumps of other processes, and
//! every other dbghelp function, are Windows' own.

use std::ffi::c_void;

use windows_sys::Win32::Foundation::{SetLastError, ERROR_NOT_SUPPORTED, HANDLE};
use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetProcessId};

pub const DBGHELP: &str = "dbghelp.dll";

type MiniDumpFn = unsafe extern "system" fn(HANDLE, u32, HANDLE, i32, *const c_void, *const c_void, *const c_void) -> i32;

/// A dump of this process is not written (see the module comment); FALSE with
/// ERROR_NOT_SUPPORTED, as for a dump type the system cannot write.
#[allow(non_snake_case)]
unsafe extern "system" fn MiniDumpWriteDump(process: HANDLE, pid: u32, file: HANDLE, ty: i32, exception: *const c_void, user: *const c_void, callback: *const c_void) -> i32 {
    let me = GetCurrentProcessId();
    if pid == me || GetProcessId(process) == me {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_warn!("dbghelp: the title's minidump of its own process is not written (an in-process dump can hang the process; XCL's log has the exception and the threads)"));
        SetLastError(ERROR_NOT_SUPPORTED);
        return 0;
    }
    let name = crate::pe::ImportName::Name { hint: 0, name: "MiniDumpWriteDump".into() };
    match crate::resolver::system_proc(DBGHELP, &name) {
        Some(f) => std::mem::transmute::<usize, MiniDumpFn>(f)(process, pid, file, ty, exception, user, callback),
        None => {
            SetLastError(ERROR_NOT_SUPPORTED);
            0
        }
    }
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![MiniDumpWriteDump]
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// What the title loads by name and asks for: `MiniDumpWriteDump` is XCL's and refuses a dump
    /// of this process at once (Windows' would write it – or hang); the rest of dbghelp is
    /// Windows' own.
    #[test]
    fn a_minidump_of_the_process_itself_is_declined() {
        let name = |s: &str| crate::pe::ImportName::Name { hint: 0, name: s.into() };
        let h = crate::resolver::load_library("dbghelp.dll").expect("LoadLibrary(dbghelp.dll)");
        let f = crate::resolver::get_proc_address(h, &name("MiniDumpWriteDump")).expect("GetProcAddress(MiniDumpWriteDump)");
        assert_eq!(f, MiniDumpWriteDump as usize);
        let f: MiniDumpFn = unsafe { std::mem::transmute(f) };
        let ok = unsafe { f(GetCurrentProcess(), GetCurrentProcessId(), std::ptr::null_mut(), 0, std::ptr::null(), std::ptr::null(), std::ptr::null()) };
        assert_eq!((ok, unsafe { GetLastError() }), (0, ERROR_NOT_SUPPORTED));
        let sym = crate::resolver::get_proc_address(h, &name("SymInitializeW")).expect("GetProcAddress(SymInitializeW)");
        assert_eq!(Some(sym), crate::resolver::system_proc(DBGHELP, &name("SymInitializeW")));
    }
}
