//! The kernel-side form of [`DeviceKeep`], the per-term truncation rule the fused layer applies on device.

pub use crate::truncation::DeviceKeep;

impl DeviceKeep {
    /// `(kind, eps, k)` as the kernels take them; kinds match `KEEP_*` in `kernels/prelude.cuh`.
    pub(crate) fn args(self) -> (u32, f64, u32) {
        match self {
            DeviceKeep::Keep => (0, 0.0, 0),
            DeviceKeep::Coeff(eps) => (1, eps, 0),
            DeviceKeep::Weight(k) => (2, 0.0, k),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::truncation::{And, ApproxTopN, CoefficientThreshold, TopN, WeightCutoff};
    use crate::TruncationPolicy;

    #[test]
    fn builtins_lower_and_finalizing_or_composed_policies_do_not() {
        let coeff = <_ as TruncationPolicy<1>>::device_policy(&CoefficientThreshold(1e-3));
        assert_eq!(coeff, Some(DeviceKeep::Coeff(1e-3)));
        let weight = <_ as TruncationPolicy<2>>::device_policy(&WeightCutoff(4));
        assert_eq!(weight, Some(DeviceKeep::Weight(4)));
        assert_eq!(<_ as TruncationPolicy<1>>::device_policy(&TopN(10)), None);
        assert_eq!(
            <_ as TruncationPolicy<1>>::device_policy(&ApproxTopN(10)),
            None
        );
        assert_eq!(
            <_ as TruncationPolicy<1>>::device_policy(&And(
                CoefficientThreshold(1e-3),
                WeightCutoff(4)
            )),
            None
        );
        assert_eq!(
            <_ as TruncationPolicy<1>>::device_policy(&crate::test_support::KeepAll),
            Some(DeviceKeep::Keep)
        );
    }
}
