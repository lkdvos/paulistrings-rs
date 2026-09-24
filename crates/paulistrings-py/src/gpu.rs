//! The CUDA surface of the bindings: `device=` parsing, the resident `GpuPauliSum` class, and `GpuError` as a Python exception.
//! `GpuPauliSum` exists in every build so the Python name is stable; without the `cuda` feature its storage enum is uninhabited, so nothing can construct one.

use crate::sum::{check_num_qubits, parse_direction, parse_engine, PauliSum, PropagationStats};
use crate::truncation_spec::{spec_has_exact_topn, PolicySpec, PyTruncation};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

#[cfg(feature = "cuda")]
pub(crate) use cuda::{gpu_error, resolve_device, GpuPauliSumImpl};
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

/// Why a spec containing an exact `topn` cannot run on a device, prefixed by the caller with the kwarg or method that asked for one.
pub(crate) const TOPN_DEVICE_MSG: &str =
    "exact truncation.topn is not available on a CUDA device; use truncation.approx_topn";

/// Why a request naming several devices is refused.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const MULTI_DEVICE_MSG: &str =
    "multi-device propagation is not available yet; pass one device ordinal";

#[cfg(feature = "cuda")]
mod cuda {
    use super::{DeviceRequest, MULTI_DEVICE_MSG};
    use crate::circuit::CircuitImpl;
    use crate::sum::{PauliSumImpl, PropagateFailure};
    use crate::truncation_spec::{PolicySpec, SpecPolicy};
    use paulistrings::gpu::{device_count, GpuError, GpuPauliSum as CoreGpuPauliSum};
    use paulistrings::{Direction, PartitionTrace, PropagateOptions};
    use pyo3::exceptions::{PyMemoryError, PyNotImplementedError, PyRuntimeError, PyValueError};
    use pyo3::PyErr;

    /// A core [`GpuError`] as a Python exception: `MemoryError` for an exhausted device, `NotImplementedError` for what the backend does not do, `RuntimeError` for the rest.
    pub(crate) fn gpu_error(err: GpuError) -> PyErr {
        match err {
            GpuError::OutOfMemory { .. } => PyMemoryError::new_err(err.to_string()),
            GpuError::Unsupported(_) => PyNotImplementedError::new_err(err.to_string()),
            GpuError::NoDevice
            | GpuError::LibraryMissing(_)
            | GpuError::Driver(_)
            | GpuError::Compile { .. }
            | GpuError::Topology(_)
            | GpuError::Poisoned { .. } => PyRuntimeError::new_err(err.to_string()),
        }
    }

    /// The one device ordinal `request` names on this machine; `shown` is the kwarg as the caller spelled it.
    pub(crate) fn resolve_device(request: &DeviceRequest, shown: &str) -> Result<u32, PyErr> {
        let count = device_count();
        if count == 0 {
            return Err(PyRuntimeError::new_err(format!(
                "{shown}: no CUDA device is visible to this process \
                 (paulistrings.cuda_available() is False)"
            )));
        }
        match request {
            DeviceRequest::Auto if count > 1 => Err(PyNotImplementedError::new_err(format!(
                "{shown} sees {count} CUDA devices: {MULTI_DEVICE_MSG}"
            ))),
            DeviceRequest::Auto => Ok(0),
            DeviceRequest::Ordinals(list) if list.len() > 1 => Err(PyNotImplementedError::new_err(
                format!("{shown}: {MULTI_DEVICE_MSG}"),
            )),
            DeviceRequest::Ordinals(list) => {
                let device = list[0];
                if device as usize >= count {
                    return Err(PyValueError::new_err(format!(
                        "{shown}, but this process sees {count} CUDA device(s)"
                    )));
                }
                Ok(device)
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
    /// Exact `truncation.topn` raises `NotImplementedError` (use `approx_topn`), as does a channel on more than two qubits other than a Pauli rotation.
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
        if spec_has_exact_topn(spec) {
            return Err(PyNotImplementedError::new_err(format!(
                "GpuPauliSum.propagate: {TOPN_DEVICE_MSG}"
            )));
        }
        let inner = &mut self.inner;
        Ok(py.allow_threads(move || inner.propagate(&circuit.inner, spec, dir, options, traced))?)
    }
}
