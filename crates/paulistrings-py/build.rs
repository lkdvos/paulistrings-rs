//! Build plumbing for the `mpi` feature of the **bindings**, and nothing else.
//!
//! Without `--features mpi` this script emits no directive at all, so the
//! default `_paulistrings` cdylib's link line is exactly what it was before
//! the script existed (CLAUDE.md §Determinism policy).
//!
//! With the feature on it re-adds one `-Wl,-rpath,<libdir>` per MPI library
//! directory, so the extension module finds `libmpi.so.40` at `import` time
//! from a shell with no `module load`. This is a **deliberate copy** of the
//! rpath half of `crates/paulistrings/build.rs`: `cargo:rustc-link-arg` is not
//! inherited from a dependency's build script, so the core crate's rpath never
//! reaches this crate's cdylib and the probe has to run again here. Keep the
//! two in sync; the core's copy additionally stamps
//! `PAULISTRINGS_MPI_BUILD_VERSION`, which only its own code reads.
//!
//! `PAULISTRINGS_MPI_RPATH=0` opts out (a distro build with the MPI libraries
//! already on the loader's default search path). Probing failures are
//! `cargo:warning=` only — rsmpi's own build script fails right after with the
//! authoritative message, and duplicating it as a hard error here would bury
//! the useful one.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MPICC");
    println!("cargo:rerun-if-env-changed=PAULISTRINGS_MPI_RPATH");

    if std::env::var_os("CARGO_FEATURE_MPI").is_none() {
        return;
    }
    if std::env::var("PAULISTRINGS_MPI_RPATH").as_deref() == Ok("0") {
        return;
    }

    let mpicc = std::env::var("MPICC").unwrap_or_else(|_| "mpicc".to_string());
    match libdirs(&mpicc) {
        Some(dirs) if !dirs.is_empty() => {
            for dir in dirs {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            }
        }
        _ => println!(
            "cargo:warning=could not get MPI library directories from `{mpicc}`, so the extension \
             module will carry no rpath and `import paulistrings` may fail to find libmpi outside \
             the build environment. On Flatiron hosts: module load modules/2.4-20250724 \
             openmpi/5.0.6 llvm/19.1.7 && export LIBCLANG_PATH=$(llvm-config --libdir). Set \
             PAULISTRINGS_MPI_RPATH=0 to silence this."
        ),
    }
}

/// `mpicc <arg>`, trimmed, or `None` if it could not be run or failed.
fn run(mpicc: &str, arg: &str) -> Option<String> {
    let out = Command::new(mpicc).arg(arg).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The MPI library directories: Open MPI's `--showme:libdirs`, falling back to
/// the `-L` entries of `mpicc -show` (MPICH, Intel MPI, and Open MPI wrappers
/// built without the `--showme` family).
fn libdirs(mpicc: &str) -> Option<Vec<String>> {
    if let Some(out) = run(mpicc, "--showme:libdirs") {
        let dirs: Vec<String> = out.split_whitespace().map(str::to_string).collect();
        if !dirs.is_empty() {
            return Some(dirs);
        }
    }
    let out = run(mpicc, "-show")?;
    Some(
        out.split_whitespace()
            .filter_map(|tok| tok.strip_prefix("-L"))
            .filter(|dir| !dir.is_empty())
            .map(str::to_string)
            .collect(),
    )
}
