//! On-disk cache for NVRTC-compiled PTX, keyed by source, compile options, NVRTC version and crate version.

use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::nvrtc::{CompileOptions, Ptx};

/// `$PAULISTRINGS_KERNEL_CACHE` (`off` disables it), else `$XDG_CACHE_HOME/paulistrings/kernels`, else `~/.cache/paulistrings/kernels`.
fn cache_dir() -> Option<PathBuf> {
    match env::var("PAULISTRINGS_KERNEL_CACHE") {
        Ok(value) if value == "off" => return None,
        Ok(value) if !value.is_empty() => return Some(PathBuf::from(value)),
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

/// NVRTC's own version, part of the key since another NVRTC can emit other PTX; `(0, 0)` without the library, a key no real compile writes.
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

fn entry_name(source: &str, options: &CompileOptions) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    options.hash(&mut hasher);
    nvrtc_version().hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    format!("{:016x}.ptx", hasher.finish())
}

/// The cached PTX for `(source, options)`; any failure (disabled, missing, unreadable, empty) is a miss, never an error, so the caller falls back to compiling.
pub(crate) fn lookup(source: &str, options: &CompileOptions) -> Option<Ptx> {
    let dir = cache_dir()?;
    let path = dir.join(entry_name(source, options));
    let text = fs::read_to_string(&path).ok()?;
    if text.is_empty() {
        return None;
    }
    Some(Ptx::from_src(text))
}

/// Best-effort write-back of `ptx_text` through a temp file and a rename, so a concurrent reader never sees a torn file; an unwritable directory is skipped with a debug line.
pub(crate) fn store(source: &str, options: &CompileOptions, ptx_text: &str) {
    let Some(dir) = cache_dir() else { return };
    if let Err(e) = fs::create_dir_all(&dir) {
        log::debug!(
            target: "paulistrings::propagate",
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
