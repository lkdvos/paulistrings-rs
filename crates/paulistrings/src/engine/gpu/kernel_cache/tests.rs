use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

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
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_second_lookup_hits_what_the_first_store_wrote() {
    let dir = TempDir::new("hit");
    let o = opts(&["-DW=2"]);
    assert!(lookup(dir.path(), "src-a", &o).is_none());
    store(dir.path(), "src-a", &o, "ptx-text-a");
    let hit = lookup(dir.path(), "src-a", &o).expect("cache hit");
    assert_eq!(hit.to_src(), "ptx-text-a");
}

#[test]
fn a_different_source_or_option_misses() {
    let dir = TempDir::new("miss");
    let dir = dir.path();
    let o = opts(&["-DW=2"]);
    store(dir, "src-a", &o, "ptx-text-a");
    assert!(lookup(dir, "src-b", &o).is_none(), "different source");
    assert!(
        lookup(dir, "src-a", &opts(&["-DW=4"])).is_none(),
        "different option"
    );
    let mut other_arch = o.clone();
    other_arch.arch = Some("compute_86");
    assert!(
        lookup(dir, "src-a", &other_arch).is_none(),
        "different arch"
    );
    let mut other_fmad = o.clone();
    other_fmad.fmad = Some(true);
    assert!(
        lookup(dir, "src-a", &other_fmad).is_none(),
        "different fmad"
    );
}

#[test]
fn the_directory_knob_reads_off_a_path_or_the_default() {
    assert_eq!(DIRECTORY.resolve(Some("off")), None);
    assert_eq!(
        DIRECTORY.resolve(Some("/scratch/kernels")),
        Some(PathBuf::from("/scratch/kernels"))
    );
    assert_eq!(
        default_directory(Some("/xdg".into()), Some("/home/me".into())),
        Some(PathBuf::from("/xdg/paulistrings/kernels"))
    );
    assert_eq!(
        default_directory(Some(String::new()), Some("/home/me".into())),
        Some(PathBuf::from("/home/me/.cache/paulistrings/kernels"))
    );
    assert_eq!(
        default_directory(None, Some("/home/me".into())),
        Some(PathBuf::from("/home/me/.cache/paulistrings/kernels"))
    );
    assert_eq!(default_directory(None, None), None);
}

#[test]
fn a_corrupt_entry_is_a_miss_not_an_error() {
    let dir = TempDir::new("corrupt");
    let dir = dir.path();
    let o = opts(&["-DW=2"]);
    store(dir, "src-a", &o, "ptx-text-a");
    fs::write(dir.join(entry_name("src-a", &o)), "").unwrap();
    assert!(lookup(dir, "src-a", &o).is_none(), "empty file is a miss");
    store(dir, "src-a", &o, "ptx-text-recompiled");
    assert_eq!(
        lookup(dir, "src-a", &o).unwrap().to_src(),
        "ptx-text-recompiled"
    );
}

#[test]
fn concurrent_writers_leave_one_valid_file() {
    let dir = TempDir::new("concurrent");
    let o = opts(&["-DW=2"]);
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let o = o.clone();
            let dir = dir.path().to_path_buf();
            thread::spawn(move || store(&dir, "src-a", &o, &format!("ptx-text-{i}")))
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let hit = lookup(dir.path(), "src-a", &o).expect("some writer's file survives intact");
    assert!(hit.to_src().starts_with("ptx-text-"));
}

/// End-to-end through `module::compile_ptx`: the second call with identical inputs must not invoke NVRTC again, and no cache always compiles.
#[test]
fn a_second_compile_with_the_same_inputs_hits_the_cache() {
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return;
    }
    use super::super::module::{compile_ptx, NVRTC_COMPILES};

    let own = ["-DPAULISTRINGS_CACHE_E2E=1".to_string()];
    let dir = TempDir::new("e2e-hit");
    let cache = Some(dir.path());
    let compiles = || NVRTC_COMPILES.with(std::cell::Cell::get);
    let before = compiles();
    compile_ptx(2, "compute_80", &own, cache).expect("first compile");
    assert_eq!(compiles(), before + 1, "first call always compiles");
    compile_ptx(2, "compute_80", &own, cache).expect("second compile");
    assert_eq!(compiles(), before + 1, "second call hits the cache");
    let other = [own[0].clone(), "-DFP_BITS=8".to_string()];
    compile_ptx(2, "compute_80", &other, cache).expect("different options");
    assert_eq!(compiles(), before + 2, "different options miss");
    compile_ptx(2, "compute_80", &own, None).expect("uncached compile");
    assert_eq!(compiles(), before + 3, "no cache never hits");
}
