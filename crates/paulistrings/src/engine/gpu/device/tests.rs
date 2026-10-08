use super::*;

#[test]
fn cuda_available_never_panics() {
    let _ = cuda_available();
}

#[cfg(feature = "mpi")]
#[test]
fn nccl_available_never_panics() {
    let _ = nccl_available();
}

/// A `libnccl.so*` file on `LD_LIBRARY_PATH` means the module is loaded; without one the test returns early.
/// A directory name containing "nccl" is not enough, since the `mpi` build script puts its `OUT_DIR` on the path.
#[cfg(feature = "mpi")]
#[test]
fn nccl_available_true_with_module_on_path() {
    let module_on_path = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .any(|dir| {
            std::fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .flatten()
                        .any(|e| e.file_name().to_string_lossy().starts_with("libnccl.so"))
                })
                .unwrap_or(false)
        });
    if !module_on_path {
        return;
    }
    crate::require_cuda!();
    assert!(nccl_available());
}

#[test]
fn devices_are_consistent() {
    crate::require_cuda!();
    let list = devices().expect("cuda_available() was true");
    assert_eq!(list.len(), device_count());
    for d in &list {
        assert!(d.compute_capability >= (5, 0));
        assert!(d.total_mem > 0);
    }
}
