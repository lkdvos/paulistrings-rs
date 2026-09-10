//! Build plumbing for the `mpi` feature, and nothing else.
//!
//! Without `--features mpi` this script emits no directive at all, so the
//! default build's codegen and link line are exactly what they were before it
//! existed (CLAUDE.md §Determinism policy: the default build must stay
//! byte-identical).
//!
//! With the feature on it does two things `rsmpi` does not:
//!
//! 1. **rpath.** `rsmpi`'s build script probes `$MPICC` for the link line but
//!    *drops* `-Wl,-rpath` entries, so a binary linked against an Lmod
//!    Open MPI runs only while that module is loaded. We re-add one
//!    `-Wl,-rpath,<libdir>` per `mpicc --showme:libdirs` entry, which is what
//!    lets `mpirun` and `srun` launch the test binary from a bare shell. Set
//!    `PAULISTRINGS_MPI_RPATH=0` to opt out (a distro build with the MPI
//!    libraries on the default search path, say).
//! 2. **A build-time version stamp.** `PAULISTRINGS_MPI_BUILD_VERSION` records
//!    what `mpicc` said at compile time; the transport compares it against the
//!    library version reported at run time and warns on a mismatch. Open MPI
//!    4.1 and 5.0 share the soname `libmpi.so.40`, so a wrong module is a
//!    silent ABI hazard rather than a link error.
//!
//! Probing failures are `cargo:warning=` only — `rsmpi`'s own build script
//! fails right after with the authoritative message, and duplicating it as a
//! hard error here would just bury the useful one.

use std::process::Command;

fn main() {
    // Cheap and unconditional: without these the script's own inputs change
    // silently.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MPICC");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=PAULISTRINGS_MPI_RPATH");

    if std::env::var_os("CARGO_FEATURE_MPI").is_none() {
        return;
    }

    let mpicc = std::env::var("MPICC").unwrap_or_else(|_| "mpicc".to_string());

    match run(&mpicc, "--showme:version") {
        Some(version) => println!("cargo:rustc-env=PAULISTRINGS_MPI_BUILD_VERSION={version}"),
        None => println!("cargo:rustc-env=PAULISTRINGS_MPI_BUILD_VERSION=unknown"),
    }

    if std::env::var("PAULISTRINGS_MPI_RPATH").as_deref() == Ok("0") {
        return;
    }
    match libdirs(&mpicc) {
        Some(dirs) if !dirs.is_empty() => {
            for dir in dirs {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            }
        }
        _ => println!(
            "cargo:warning=could not get MPI library directories from `{mpicc}`, so the binary \
             will carry no rpath and may not run outside the build environment. On Flatiron \
             hosts: module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 && export \
             LIBCLANG_PATH=$(llvm-config --libdir). Set PAULISTRINGS_MPI_RPATH=0 to silence this."
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
