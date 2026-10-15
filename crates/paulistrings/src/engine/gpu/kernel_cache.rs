//! On-disk cache for NVRTC-compiled PTX, keyed by source, compile options, NVRTC version and crate version.

use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::nvrtc::{CompileOptions, Ptx};

use super::GPU_LOG_TARGET;
use crate::engine::cuda_context::libnvrtc_present;
use crate::engine::knob::Knob;

/// `PAULISTRINGS_KERNEL_CACHE`: `off` disables the cache, any other value is its directory, unset is [`default_directory`].
static DIRECTORY: Knob<Option<PathBuf>> = Knob::new(
    "PAULISTRINGS_KERNEL_CACHE",
    GPU_LOG_TARGET,
    |text| Some((text != "off").then(|| PathBuf::from(text))),
    || default_directory(env::var("XDG_CACHE_HOME").ok(), env::var("HOME").ok()),
);

/// The process's cache directory, `None` when disabled.
pub(crate) fn directory() -> Option<&'static Path> {
    DIRECTORY.get().as_deref()
}

/// `$XDG_CACHE_HOME/paulistrings/kernels`, else `$HOME/.cache/paulistrings/kernels`.
fn default_directory(xdg_cache_home: Option<String>, home: Option<String>) -> Option<PathBuf> {
    let base = match xdg_cache_home.filter(|xdg| !xdg.is_empty()) {
        Some(xdg) => PathBuf::from(xdg),
        None => PathBuf::from(home?).join(".cache"),
    };
    Some(base.join("paulistrings").join("kernels"))
}

/// NVRTC's own version, part of the key since another NVRTC can emit other PTX; `(0, 0)` without the library, a key no real compile writes.
fn nvrtc_version() -> (i32, i32) {
    if !libnvrtc_present() {
        return (0, 0);
    }
    let mut major = 0;
    let mut minor = 0;
    unsafe {
        cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor);
    }
    (major, minor)
}

fn entry_name(source: &str, options: &CompileOptions) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    options.hash(&mut hasher);
    nvrtc_version().hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    format!("{:016x}.ptx", hasher.finish())
}

/// The PTX cached in `dir` for `(source, options)`; any failure (missing, unreadable, empty) is a miss, never an error, so the caller falls back to compiling.
pub(crate) fn lookup(dir: &Path, source: &str, options: &CompileOptions) -> Option<Ptx> {
    let path = dir.join(entry_name(source, options));
    let text = fs::read_to_string(&path).ok()?;
    if text.is_empty() {
        return None;
    }
    Some(Ptx::from_src(text))
}

/// Best-effort write-back of `ptx_text` into `dir` through a temp file and a rename, so a concurrent reader never sees a torn file; an unwritable directory is skipped with a debug line.
pub(crate) fn store(dir: &Path, source: &str, options: &CompileOptions, ptx_text: &str) {
    if let Err(e) = fs::create_dir_all(dir) {
        log::debug!(
            target: GPU_LOG_TARGET,
            "kernel cache directory {} unwritable, compiling without a cache: {e}",
            dir.display()
        );
        return;
    }
    let path = dir.join(entry_name(source, options));
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let temporary = dir.join(format!(".tmp-{}-{unique}", std::process::id()));
    if fs::write(&temporary, ptx_text).is_err() {
        let _ = fs::remove_file(&temporary);
        return;
    }
    if fs::rename(&temporary, &path).is_err() {
        let _ = fs::remove_file(&temporary);
    }
}

#[cfg(test)]
mod tests;
