//! The CUDA surface of the bindings: `device=` parsing, the resident `GpuPauliSum` class, and `GpuError` as a Python exception.
//! `GpuPauliSum` exists in every build so the Python name is stable; without the `cuda` feature its storage enum is uninhabited, so nothing can construct one.

use crate::sum::{check_num_qubits, parse_direction, parse_engine, PauliSum, PropagationStats};
use crate::truncation_spec::{PolicySpec, PyTruncation};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

#[cfg(all(feature = "cuda", feature = "mpi"))]
pub(crate) use cuda::resolve_rank_device;
#[cfg(feature = "cuda")]
pub(crate) use cuda::{gpu_error, resolve_device, resolve_devices, GpuPauliSumImpl};
#[cfg(not(feature = "cuda"))]
pub(crate) use no_cuda::{cuda_unavailable_error, GpuPauliSumImpl};

/// `value` as a device ordinal, or a `ValueError` naming it.
fn ordinal(value: i64) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "device={value} is not a CUDA device ordinal (0, 1, ...)"
        ))
    })
}

/// A `device=` value as spelled, before it is checked against the visible devices.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) enum DeviceRequest {
    Auto,
    Ordinals(Vec<u32>),
}

/// `device=` → a [`DeviceRequest`]: an `int` ordinal, a non-empty `list[int]`, or `"auto"`. The caller has already mapped `None` to "no device".
pub(crate) fn parse_device(obj: &Bound<'_, PyAny>) -> PyResult<DeviceRequest> {
    if let Ok(name) = obj.extract::<String>() {
        return match name.as_str() {
            "auto" => Ok(DeviceRequest::Auto),
            other => Err(PyValueError::new_err(format!(
                "device must be an int, a list of ints, or 'auto', got {other:?}"
            ))),
        };
    }
    // `bool` is an `int` subclass, so `device=True` would otherwise mean device 1.
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "device must be None, an int, a list of ints, or 'auto', not a bool",
        ));
    }
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(DeviceRequest::Ordinals(vec![ordinal(value)?]));
    }
    if let Ok(values) = obj.extract::<Vec<i64>>() {
        if values.is_empty() {
            return Err(PyValueError::new_err(
                "device=[]: pass at least one device ordinal, or None for the host engine",
            ));
        }
        return Ok(DeviceRequest::Ordinals(
            values.into_iter().map(ordinal).collect::<PyResult<_>>()?,
        ));
    }
    Err(PyTypeError::new_err(
        "device must be None, an int, a list of ints, or 'auto'",
    ))
}

/// Why a spec containing an exact `topn` cannot run on more than one CUDA device (or one device per rank under `comm=`), prefixed by the caller with the kwarg or method that asked for one.
/// A lone device (`device=<int>`, `to_device`, or `device=` resolving to one ordinal) supports it directly.
pub(crate) const TOPN_DEVICE_MSG: &str =
    "exact truncation.topn has no collective form above one CUDA device; use truncation.approx_topn, or a single device= ordinal";

/// The `RuntimeError` for a `device=` that pairs with `comm=` in a build lacking one of the two features.
#[cfg(not(all(feature = "cuda", feature = "mpi")))]
pub(crate) fn device_comm_unavailable_error() -> PyErr {
    let missing = match (cfg!(feature = "cuda"), cfg!(feature = "mpi")) {
        (false, false) => "the cuda and mpi features",
        (false, true) => "the cuda feature",
        _ => "the mpi feature",
    };
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "device= with comm= runs one CUDA device per MPI rank, which needs the extension built \
         with both the cuda and mpi features; this build lacks {missing} \
         (`maturin develop --release --features cuda,mpi`)"
    ))
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::DeviceRequest;
    use crate::circuit::CircuitImpl;
    use crate::sum::{PauliSumImpl, PropagateFailure};
    use crate::truncation_spec::{PolicySpec, SpecPolicy};
    use paulistrings::bucket::P_MAX_BITS;
    use paulistrings::gpu::{device_count, GpuError, GpuPauliSum as CoreGpuPauliSum};
    use paulistrings::{Direction, PartitionTrace, PropagateOptions};
    use pyo3::exceptions::{PyMemoryError, PyNotImplementedError, PyRuntimeError, PyValueError};
    use pyo3::PyErr;

    /// The `RuntimeError` text for a peer's [`GpuError::Poisoned`], a plain function so it is testable without linking Python.
    pub(crate) fn poisoned_message(rank: usize, layer: usize) -> String {
        format!(
            "CUDA device partition {rank} (the MPI rank under comm=) failed at layer {layer}, so \
             every partition abandoned this propagate; the failing partition raises its own error"
        )
    }

    /// A core [`GpuError`] as a Python exception: `MemoryError` for an exhausted device, `NotImplementedError` for what the backend does not do, `ValueError` for a placement that does not resolve (`OSError` if a syscall failed), `RuntimeError` for the rest.
    pub(crate) fn gpu_error(err: GpuError) -> PyErr {
        match err {
            GpuError::OutOfMemory { .. } => PyMemoryError::new_err(err.to_string()),
            GpuError::Unsupported(_) => PyNotImplementedError::new_err(err.to_string()),
            GpuError::Topology(err) => crate::sum::topology_error(err),
            GpuError::Poisoned { rank, layer } => {
                PyRuntimeError::new_err(poisoned_message(rank, layer))
            }
            GpuError::NoDevice
            | GpuError::LibraryMissing(_)
            | GpuError::Driver(_)
            | GpuError::Compile { .. } => PyRuntimeError::new_err(err.to_string()),
            #[cfg(feature = "nccl")]
            GpuError::Nccl { .. } | GpuError::Timeout { .. } => {
                PyRuntimeError::new_err(err.to_string())
            }
        }
    }

    /// The number of visible devices, or the `RuntimeError` for none.
    fn visible_devices(shown: &str) -> Result<usize, PyErr> {
        match device_count() {
            0 => Err(PyRuntimeError::new_err(format!(
                "{shown}: no CUDA device is visible to this process \
                 (paulistrings.cuda_available() is False)"
            ))),
            count => Ok(count),
        }
    }

    /// `device` if this process can see it, else a `ValueError`.
    fn check_ordinal(device: u32, count: usize, shown: &str) -> Result<u32, PyErr> {
        if device as usize >= count {
            return Err(PyValueError::new_err(format!(
                "{shown} names device {device}, but this process sees {count} CUDA device(s)"
            )));
        }
        Ok(device)
    }

    /// The devices `request` places one partition on each, in partition order; `shown` is the kwarg as the caller spelled it.
    /// `"auto"` takes devices `0..k` for the largest power of two `k` visible, as `partitions="auto"` rounds its node count.
    pub(crate) fn resolve_devices(request: &DeviceRequest, shown: &str) -> Result<Vec<u32>, PyErr> {
        let count = visible_devices(shown)?;
        let max = 1usize << P_MAX_BITS;
        let devices: Vec<u32> = match request {
            DeviceRequest::Auto => {
                let k = 1usize << count.min(max).ilog2();
                (0..k as u32).collect()
            }
            DeviceRequest::Ordinals(list) => list
                .iter()
                .map(|&device| check_ordinal(device, count, shown))
                .collect::<Result<_, _>>()?,
        };
        let n = devices.len();
        if !n.is_power_of_two() {
            return Err(PyValueError::new_err(format!(
                "{shown} names {n} devices, which is not a power of two: a multi-device run \
                 places one partition per listed device and a partition is named by log2(P) \
                 GF(2) rows, so the list must have 1, 2, 4, 8, ... entries (repeat an ordinal to \
                 put several partitions on one device)"
            )));
        }
        if n > max {
            return Err(PyValueError::new_err(format!(
                "{shown} names {n} devices, but a run has at most {max} partitions"
            )));
        }
        Ok(devices)
    }

    /// The one device ordinal `request` names on this machine, for `to_device`.
    pub(crate) fn resolve_device(request: &DeviceRequest, shown: &str) -> Result<u32, PyErr> {
        Ok(resolve_devices(request, shown)?[0])
    }

    /// This rank's device under `comm=`: an ordinal it can see, or `auto` (the group's `local_device_for_comm` pick) for `"auto"`. Local, so the caller agrees the outcome over the group.
    #[cfg(feature = "mpi")]
    pub(crate) fn resolve_rank_device(
        request: &DeviceRequest,
        shown: &str,
        rank: u32,
        auto: Result<u32, paulistrings::gpu::GpuError>,
    ) -> Result<u32, PyErr> {
        let count = visible_devices(&format!("{shown} on rank {rank}"))?;
        match request {
            DeviceRequest::Auto => auto.map_err(gpu_error),
            DeviceRequest::Ordinals(list) => {
                check_ordinal(list[0], count, &format!("{shown} on rank {rank}"))
            }
        }
    }

    /// Width-dispatch enum over the device-resident core sum, the device twin of `PauliSumImpl`.
    pub(crate) enum GpuPauliSumImpl {
        W1(CoreGpuPauliSum<1>),
        W2(CoreGpuPauliSum<2>),
        W4(CoreGpuPauliSum<4>),
        W8(CoreGpuPauliSum<8>),
        W16(CoreGpuPauliSum<16>),
    }

    impl GpuPauliSumImpl {
        pub(crate) fn upload(sum: &PauliSumImpl, device: u32) -> Result<Self, GpuError> {
            Ok(
                for_each_width_convert!(PauliSumImpl => GpuPauliSumImpl, sum, |s| {
                    CoreGpuPauliSum::from_host(s, device)?
                }),
            )
        }

        pub(crate) fn download(&self) -> Result<PauliSumImpl, GpuError> {
            Ok(
                for_each_width_convert!(GpuPauliSumImpl => PauliSumImpl, self, |s| {
                    s.to_host()?
                }),
            )
        }

        pub(crate) fn len(&self) -> usize {
            for_each_width!(self, |s| s.len())
        }

        pub(crate) fn num_qubits(&self) -> usize {
            for_each_width!(self, |s| s.num_qubits())
        }

        pub(crate) fn device(&self) -> u32 {
            for_each_width!(self, |s| s.device())
        }

        pub(crate) fn num_buckets(&self) -> usize {
            for_each_width!(self, |s| 1usize << s.bits())
        }

        /// Step the resident sum through `circuit`, returning this call's trace when `traced`.
        /// The core trace is drained after every call, so a sum stepped many times never accumulates records.
        pub(crate) fn propagate(
            &mut self,
            circuit: &CircuitImpl,
            spec: &PolicySpec,
            direction: Direction,
            options: PropagateOptions,
            traced: bool,
        ) -> Result<Option<PartitionTrace>, PropagateFailure> {
            for_each_width_propagate!(
                GpuPauliSumImpl,
                self,
                circuit,
                |s, c, W, _wrap| {
                    if traced {
                        s.enable_trace();
                        let _ = s.take_trace();
                    }
                    let result =
                        s.propagate_with_options(c, &SpecPolicy::<W>(spec), direction, options);
                    let trace = s.take_trace();
                    result.map_err(PropagateFailure::Gpu)?;
                    Ok(if traced { trace } else { None })
                },
                else Err(PropagateFailure::WidthMismatch)
            )
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod no_cuda {
    use crate::circuit::CircuitImpl;
    use crate::sum::{PauliSumImpl, PropagateFailure};
    use crate::truncation_spec::PolicySpec;
    use paulistrings::{Direction, PartitionTrace, PropagateOptions};
    use pyo3::PyErr;

    /// The `RuntimeError` a `device=` or `to_device` raises in a build without the `cuda` feature.
    pub(crate) fn cuda_unavailable_error() -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(
            "paulistrings was built without CUDA support; rebuild with \
             `maturin develop --release --features cuda`",
        )
    }

    /// Uninhabited: a build without the `cuda` feature has no device sum to hold.
    pub(crate) enum GpuPauliSumImpl {}

    impl GpuPauliSumImpl {
        pub(crate) fn download(&self) -> Result<PauliSumImpl, PyErr> {
            match *self {}
        }

        pub(crate) fn len(&self) -> usize {
            match *self {}
        }

        pub(crate) fn num_qubits(&self) -> usize {
            match *self {}
        }

        pub(crate) fn device(&self) -> u32 {
            match *self {}
        }

        pub(crate) fn num_buckets(&self) -> usize {
            match *self {}
        }

        pub(crate) fn propagate(
            &mut self,
            _circuit: &CircuitImpl,
            _spec: &PolicySpec,
            _direction: Direction,
            _options: PropagateOptions,
            _traced: bool,
        ) -> Result<Option<PartitionTrace>, PropagateFailure> {
            match *self {}
        }
    }
}

/// `PauliSum.to_device`'s body: upload `sum` to one device, with the GIL released for the copy.
pub(crate) fn to_device(
    py: Python<'_>,
    sum: &crate::sum::PauliSumImpl,
    device: i64,
) -> PyResult<GpuPauliSum> {
    let request = DeviceRequest::Ordinals(vec![ordinal(device)?]);
    #[cfg(feature = "cuda")]
    {
        let ordinal = resolve_device(&request, &format!("device={device}"))?;
        let inner = py
            .allow_threads(|| GpuPauliSumImpl::upload(sum, ordinal))
            .map_err(gpu_error)?;
        Ok(GpuPauliSum { inner })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, request, sum);
        Err(cuda_unavailable_error())
    }
}

/// A `PauliSum` resident on one CUDA device, stepped in place by `propagate` and read back by `to_host`.
///
/// Built by `PauliSum.to_device(device)`; there is no public constructor.
/// Keeping the sum on the device between calls skips the upload and download a `PauliSum.propagate(device=...)` call pays each time, which is what a Trotter loop of many short calls wants.
/// Without the `cuda` feature the class exists but no instance can be made.
///
/// ```python
/// resident = observable.to_device(0)
/// for _ in range(steps):
///     resident.propagate(step, truncation.approx_topn(10_000_000), direction="heisenberg")
/// evolved = resident.to_host()
/// ```
#[pyclass(module = "paulistrings._paulistrings", name = "GpuPauliSum")]
pub struct GpuPauliSum {
    inner: GpuPauliSumImpl,
}

#[pymethods]
impl GpuPauliSum {
    /// Propagate the resident sum through `circuit`, in place.
    ///
    /// Arguments are `PauliSum.propagate`'s of the same name; `target_bucket_len` and `min_buckets` drive the host-side bucket schedule the device refines on top of.
    /// A `GpuPauliSum` is always one device, so exact `truncation.topn` runs here (unlike `device=[...]` or `comm=` with `device=`, which raise `NotImplementedError`); a channel on more than two qubits other than a Pauli rotation still raises `NotImplementedError`.
    /// A device error mid-run leaves the sum holding the last completed layer's output, and a later call resumes from it; an exhausted device raises `MemoryError`.
    /// The GIL is released for the duration.
    #[pyo3(signature = (circuit, policy=None, direction=None, target_bucket_len=None, min_buckets=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: Option<usize>,
        min_buckets: Option<usize>,
    ) -> PyResult<()> {
        self.step(
            py,
            circuit,
            policy,
            direction,
            target_bucket_len,
            min_buckets,
            false,
        )?;
        Ok(())
    }

    /// `propagate`, returning a `PropagationStats` for this call's layers.
    /// Its `partition` is filled as for a one-partition run, with `partition.devices` naming the device.
    #[pyo3(signature = (circuit, policy=None, direction=None, target_bucket_len=None, min_buckets=None))]
    fn propagate_with_stats(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: Option<usize>,
        min_buckets: Option<usize>,
    ) -> PyResult<PropagationStats> {
        let trace = self
            .step(
                py,
                circuit,
                policy,
                direction,
                target_bucket_len,
                min_buckets,
                true,
            )?
            .unwrap_or_default();
        Ok(PropagationStats::from_device_trace(
            &trace,
            self.inner.device(),
            self.inner.len(),
        ))
    }

    /// Download the sum as a host `PauliSum`, each bucket in the host's canonical order. The resident sum is left in place.
    fn to_host(&self, py: Python<'_>) -> PyResult<PauliSum> {
        #[cfg(feature = "cuda")]
        let inner = py
            .allow_threads(|| self.inner.download())
            .map_err(gpu_error)?;
        #[cfg(not(feature = "cuda"))]
        let inner = {
            let _ = py;
            self.inner.download()?
        };
        Ok(PauliSum { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// The CUDA device ordinal the sum lives on.
    #[getter]
    fn device(&self) -> u32 {
        self.inner.device()
    }

    /// Current bucket count of the device partition, grow-only like `PauliSum.num_buckets`.
    #[getter]
    fn num_buckets(&self) -> usize {
        self.inner.num_buckets()
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuPauliSum(num_qubits={}, terms={}, device={})",
            self.inner.num_qubits(),
            self.inner.len(),
            self.inner.device()
        )
    }
}

impl GpuPauliSum {
    /// The shared body of `propagate` and `propagate_with_stats`: every check raises before the GIL is released.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: Option<usize>,
        min_buckets: Option<usize>,
        traced: bool,
    ) -> PyResult<Option<paulistrings::PartitionTrace>> {
        let dir = parse_direction(direction)?;
        let options = parse_engine(None, None, target_bucket_len, min_buckets)?;
        check_num_qubits("GpuPauliSum", self.inner.num_qubits(), circuit)?;
        let no_op = PolicySpec::NoOp;
        let spec = policy.map_or(&no_op, |p| &p.spec);
        let inner = &mut self.inner;
        Ok(py.allow_threads(move || inner.propagate(&circuit.inner, spec, dir, options, traced))?)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    #[test]
    fn a_poisoned_peer_names_the_failing_rank_and_layer() {
        let msg = super::cuda::poisoned_message(3, 7);
        assert!(msg.contains("partition 3"), "{msg}");
        assert!(msg.contains("layer 7"), "{msg}");
    }
}
