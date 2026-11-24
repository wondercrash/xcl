//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

/// "usage/budget MB (+ reservation, when there is one)" of one segment group.
fn segment_text(s: &crate::hle::d3d12x::real::VideoSegment) -> String {
    let mut t = format!("{}/{} MB", s.usage >> 20, s.budget >> 20);
    if s.reservation != 0 {
        t += &format!(" (reserved {} MB)", s.reservation >> 20);
    }
    t
}

/// The memory line's video memory part: usage/budget of both segment groups.
fn video_memory_text(v: &crate::hle::d3d12x::real::VideoMemory) -> String {
    format!("{} local, {} non-local (usage/budget)", segment_text(&v.local), segment_text(&v.non_local))
}

/// Notices in the periodic video-memory samples: local usage first past 90 % of its budget,
/// and a budget (either group) that moved more than 5 % from the last one reported.
#[derive(Default)]
struct BudgetWatch {
    seen: Option<(u64, u64)>,
    warned_90: bool,
}

impl BudgetWatch {
    fn observe(&mut self, v: &crate::hle::d3d12x::real::VideoMemory, t: f64) -> Vec<String> {
        let mut out = Vec::new();
        let now = (v.local.budget, v.non_local.budget);
        if v.local.budget != 0 && !self.warned_90 && v.local.usage as u128 * 10 > v.local.budget as u128 * 9 {
            self.warned_90 = true;
            out.push(format!(
                "video memory: local usage {} MB passed 90% of its budget {} MB at {t:.1} s",
                v.local.usage >> 20,
                v.local.budget >> 20
            ));
        }
        match self.seen {
            None => self.seen = Some(now),
            Some((l, n)) => {
                let moved = |old: u64, new: u64| old.abs_diff(new) as u128 * 20 > old as u128;
                let (lm, nm) = (moved(l, now.0), moved(n, now.1));
                if lm || nm {
                    out.push(format!(
                        "video memory: budget changed at {t:.1} s: local {} -> {} MB, non-local {} -> {} MB",
                        l >> 20,
                        now.0 >> 20,
                        n >> 20,
                        now.1 >> 20
                    ));
                    self.seen = Some(now);
                }
            }
        }
        out
    }
}

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::xlog_info!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_info!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    // `XCL_ACTIVITY=1`: every 2 s what the title did since (a timeline of where it stops)
    if std::env::var_os("XCL_ACTIVITY").is_some() {
        let _ = std::thread::Builder::new().name("xcl-activity".into()).spawn(|| {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail]
            };
            let mut last = snap();
            let mut budget = BudgetWatch::default();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents, {} submits, {} draws, {} dispatches, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]
                );
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let private = unsafe {
                    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
                    let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
                    c.cb = std::mem::size_of_val(&c) as u32;
                    K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
                    c.PrivateUsage
                };
                let video = crate::hle::d3d12x::real::video_memory();
                if let Some(v) = &video {
                    for line in budget.observe(v, crate::log::uptime()) {
                        crate::xlog_info!("{line}");
                    }
                }
                let video = video.unwrap_or_default();
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let mb = |b: u64| b >> 20;
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {}; target pages {} MB (pools {} MB); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped)",
                    mb(private as u64),
                    video_memory_text(&video),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64)
                );
                last = now;
            }
        });
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title)
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
            }
        });
    }
    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(0);
        });
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    crate::vfs::init(crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts).own_folder(&crate::gpu::recomp::dxc::cache_dir()));
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    HOST_MODE.store(true, Ordering::Release);

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::report::write_final(None);
    crate::log::flush();
    code as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hle::d3d12x::real::{VideoMemory, VideoSegment};

    const MB: u64 = 1 << 20;
    fn seg(usage: u64, budget: u64, reservation: u64) -> VideoSegment {
        VideoSegment { usage: usage * MB, budget: budget * MB, reservation: reservation * MB }
    }

    #[test]
    fn the_memory_line_shows_usage_over_budget() {
        let v = VideoMemory { local: seg(15993, 16384, 0), non_local: seg(4829, 24000, 512) };
        assert_eq!(video_memory_text(&v), "15993/16384 MB local, 4829/24000 MB (reserved 512 MB) non-local (usage/budget)");
    }

    #[test]
    fn ninety_percent_once_and_budget_moves_over_five_percent() {
        let mut w = BudgetWatch::default();
        let at = |l: u64, b: u64| VideoMemory { local: seg(l, b, 0), non_local: seg(0, 24000, 0) };
        assert!(w.observe(&at(1000, 16000), 1.0).is_empty());
        let o = w.observe(&at(14500, 16000), 2.0);
        assert_eq!(o.len(), 1, "{o:?}");
        assert!(o[0].contains("90%") && o[0].contains("2.0 s"), "{o:?}");
        assert!(w.observe(&at(15000, 16000), 3.0).is_empty(), "90 % is reported once");
        assert!(w.observe(&at(15000, 15500), 4.0).is_empty(), "3 % is below the threshold");
        let o = w.observe(&at(15000, 12000), 5.0);
        assert_eq!(o.len(), 1, "{o:?}");
        assert!(o[0].contains("local 16000 -> 12000 MB"), "{o:?}");
    }
}
