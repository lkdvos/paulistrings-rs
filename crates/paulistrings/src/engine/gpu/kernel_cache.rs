//! On-disk cache for NVRTC-compiled PTX, keyed by source, compile options, NVRTC version and crate version.
//! See `research/FINDINGS.md` "GPU single-device profile", opportunity 3: NVRTC regenerates PTX from source on every process, though the driver's own JIT cache (`~/.nv/ComputeCache`) already hits on the PTX text.

use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::nvrtc::{CompileOptions, Ptx};

/// `$PAULISTRINGS_KERNEL_CACHE` (`off` disables it), else `$XDG_CACHE_HOME/paulistrings/kernels`, else `~/.cache/paulistrings/kernels`.
fn cache_dir() -> Option<PathBuf> {
    match env::var("PAULISTRINGS_KERNEL_CACHE") {
        Ok(v) if v == "off" => return None,
        Ok(v) if !v.is_empty() => return Some(PathBuf::from(v)),
        _ => {}
    }
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join("paulistrings").join("kernels"));
        }
    }
    let home = env::var("HOME").ok()?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("paulistrings")
            .join("kernels"),
    )
}

/// NVRTC's own version, part of the cache key: a different NVRTC can emit different PTX for the same source and options.
/// `(0, 0)` when the library is not loadable, which folds into a key that never matches a real compile's key, so it is
/// effectively cache-off rather than a wrong hit; every real caller has already checked `is_culib_present` before this runs.
fn nvrtc_version() -> (i32, i32) {
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return (0, 0);
    }
    let mut major = 0;
    let mut minor = 0;
    unsafe {
        cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor);
    }
    (major, minor)
}

fn entry_name(src: &str, opts: &CompileOptions) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut hasher);
    opts.hash(&mut hasher);
    nvrtc_version().hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    format!("{:016x}.ptx", hasher.finish())
}

/// The cached PTX for `(src, opts)`; any failure (disabled, missing, unreadable, empty) is a miss, never an error, so the caller falls back to compiling.
pub(crate) fn lookup(src: &str, opts: &CompileOptions) -> Option<Ptx> {
    let dir = cache_dir()?;
    let path = dir.join(entry_name(src, opts));
    let text = fs::read_to_string(&path).ok()?;
    if text.is_empty() {
        return None;
    }
    Some(Ptx::from_src(text))
}

/// Best-effort write-back of a freshly compiled `ptx_text`: a temp file in the same directory plus a rename, so a concurrent reader never observes a torn file.
/// An unwritable directory is skipped with one debug log line, never an error — the cache is an optimization, not a correctness requirement.
pub(crate) fn store(src: &str, opts: &CompileOptions, ptx_text: &str) {
    let Some(dir) = cache_dir() else { return };
    if let Err(e) = fs::create_dir_all(&dir) {
        log::debug!(
            target: "paulistrings::propagate",
            "kernel cache directory {} unwritable, compiling without a cache: {e}",
            dir.display()
        );
        return;
    }
    let path = dir.join(entry_name(src, opts));
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let tmp = dir.join(format!(".tmp-{}-{unique}", std::process::id()));
    if fs::write(&tmp, ptx_text).is_err() {
        let _ = fs::remove_file(&tmp);
        return;
    }
    if fs::rename(&tmp, &path).is_err() {
        let _ = fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::thread;

    /// Serializes this module's own tests around the process-global env var; other tests never touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn opts(extra: &[&str]) -> CompileOptions {
        CompileOptions {
            arch: Some("compute_80"),
            fmad: Some(false),
            options: extra.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = env::temp_dir().join(format!(
                "paulistrings-kernel-cache-test-{tag}-{}-{n}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// The env var is process-global, so every test that touches it holds `ENV_LOCK` for its whole body.
    fn with_cache_dir<T>(dir: &str, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = env::var("PAULISTRINGS_KERNEL_CACHE").ok();
        unsafe {
            env::set_var("PAULISTRINGS_KERNEL_CACHE", dir);
        }
        let result = f();
        unsafe {
            match prev {
                Some(v) => env::set_var("PAULISTRINGS_KERNEL_CACHE", v),
                None => env::remove_var("PAULISTRINGS_KERNEL_CACHE"),
            }
        }
        result
    }

    #[test]
    fn a_second_lookup_hits_what_the_first_store_wrote() {
        let dir = TempDir::new("hit");
        with_cache_dir(dir.path(), || {
            let o = opts(&["-DW=2"]);
            assert!(lookup("src-a", &o).is_none());
            store("src-a", &o, "ptx-text-a");
            let hit = lookup("src-a", &o).expect("cache hit");
            assert_eq!(hit.to_src(), "ptx-text-a");
        });
    }

    #[test]
    fn a_different_source_or_option_misses() {
        let dir = TempDir::new("miss");
        with_cache_dir(dir.path(), || {
            let o = opts(&["-DW=2"]);
            store("src-a", &o, "ptx-text-a");
            assert!(lookup("src-b", &o).is_none(), "different source");
            assert!(
                lookup("src-a", &opts(&["-DW=4"])).is_none(),
                "different option"
            );
            let mut other_arch = o.clone();
            other_arch.arch = Some("compute_86");
            assert!(lookup("src-a", &other_arch).is_none(), "different arch");
            let mut other_fmad = o.clone();
            other_fmad.fmad = Some(true);
            assert!(lookup("src-a", &other_fmad).is_none(), "different fmad");
        });
    }

    #[test]
    fn off_bypasses_the_cache_entirely() {
        let dir = TempDir::new("off");
        with_cache_dir(dir.path(), || {
            let o = opts(&["-DW=2"]);
            store("src-a", &o, "ptx-text-a");
            assert!(lookup("src-a", &o).is_some());
        });
        with_cache_dir("off", || {
            let o = opts(&["-DW=2"]);
            assert!(lookup("src-a", &o).is_none(), "off never hits");
            store("src-a", &o, "ptx-text-a");
        });
        // the "off" store must not have landed in the real temp dir either.
        with_cache_dir(dir.path(), || {
            assert!(lookup("src-a", &opts(&["-DW=2"])).is_some());
        });
    }

    #[test]
    fn a_corrupt_entry_is_a_miss_not_an_error() {
        let dir = TempDir::new("corrupt");
        with_cache_dir(dir.path(), || {
            let o = opts(&["-DW=2"]);
            store("src-a", &o, "ptx-text-a");
            let path = cache_dir().unwrap().join(entry_name("src-a", &o));
            fs::write(&path, "").unwrap();
            assert!(lookup("src-a", &o).is_none(), "empty file is a miss");
            store("src-a", &o, "ptx-text-recompiled");
            assert_eq!(lookup("src-a", &o).unwrap().to_src(), "ptx-text-recompiled");
        });
    }

    #[test]
    fn concurrent_writers_leave_one_valid_file() {
        let dir = TempDir::new("concurrent");
        with_cache_dir(dir.path(), || {
            let o = opts(&["-DW=2"]);
            // The env var is already set for the whole test by `with_cache_dir`
            // above and inherited by these threads; they must not call it again
            // (it is not reentrant and they would deadlock against this thread).
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let o = o.clone();
                    thread::spawn(move || {
                        store("src-a", &o, &format!("ptx-text-{i}"));
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            let hit = lookup("src-a", &o).expect("some writer's file survives intact");
            assert!(hit.to_src().starts_with("ptx-text-"));
        });
    }

    /// End-to-end through `module::compile_ptx`: the second call with identical inputs must not invoke NVRTC again.
    #[test]
    fn a_second_compile_with_the_same_inputs_hits_the_cache() {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return;
        }
        use super::super::module;

        // An option no other test passes, so a concurrent compile cannot write this key into the directory first.
        let own = ["-DPAULISTRINGS_CACHE_E2E=1".to_string()];
        let dir = TempDir::new("e2e-hit");
        with_cache_dir(dir.path(), || {
            let before = module::NVRTC_COMPILES.with(std::cell::Cell::get);
            module::compile_ptx(2, "compute_80", &own).expect("first compile");
            let after_first = module::NVRTC_COMPILES.with(std::cell::Cell::get);
            assert_eq!(after_first, before + 1, "first call always compiles");

            module::compile_ptx(2, "compute_80", &own).expect("second compile");
            let after_second = module::NVRTC_COMPILES.with(std::cell::Cell::get);
            assert_eq!(after_second, after_first, "second call hits the cache");

            module::compile_ptx(
                2,
                "compute_80",
                &[own[0].clone(), "-DFP_BITS=8".to_string()],
            )
            .expect("third compile, different options");
            let after_third = module::NVRTC_COMPILES.with(std::cell::Cell::get);
            assert_eq!(after_third, after_second + 1, "different options miss");
        });
    }
}
