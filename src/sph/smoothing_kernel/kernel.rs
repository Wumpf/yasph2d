use crate::units::{Real, Vector};

/// SPH smoothing kernel
///
/// Only radially symmetric kernels are supported.
/// Assume support only within smoothing length, i.e. for |r|>h user should assume 0 as result.
pub trait Kernel {
    const DIVISION_EPSILON: Real = 1.0e-10;

    /// Evaluates the kernel function for a given square of distance r_sq
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn evaluate(&self, r_sq: Real, r: Real) -> Real;

    /// Evaluates the gradient of the kernel, i.e. the first derivative for a given distance r/r_sq
    /// `ri_to_rj`: Vector from a position i to a position j, so rj - ri. Not normalized!
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn gradient(&self, ri_to_rj: Vector, r_sq: Real, r: Real) -> Vector;

    /// Evaluates the laplacian of the kernel, i.e. the second derivative.
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn laplacian(&self, r_sq: Real, r: Real) -> Real;
}

macro_rules! generate_kernel_tests {
    ($kernel_type:ident) => {
        #[cfg(test)]
        mod tests {
            use super::*;

            pub static TEST_SMOOTHING_LENGTHS: [Real; 3] = [0.5, 1.0, 123.0];

            fn run_for_different_kernel_sizes(func: impl Fn($kernel_type, Real)) {
                for &smoothing_length in TEST_SMOOTHING_LENGTHS.iter() {
                    func(super::super::$kernel_type::new(smoothing_length), smoothing_length);
                }
            }

            #[test]
            fn is_positive_within_smoothing_length() {
                run_for_different_kernel_sizes(|kernel, smoothing_length| {
                    for i in 0..100 {
                        let r = smoothing_length * (i as Real) / 100.0;
                        assert_ge!(
                            kernel.evaluate(r * r, r),
                            0.0,
                            "kernel with smoothing_length {} is smaller than zero for {}",
                            smoothing_length,
                            r
                        );
                    }
                });
            }

            #[test]
            fn is_zero_outside_of_smoothing_length() {
                run_for_different_kernel_sizes(|kernel, smoothing_length| {
                    for i in 0..100 {
                        let r = smoothing_length * (1.0000001 + (i as Real) / 10.0);
                        assert_eq!(
                            kernel.evaluate(r * r, r),
                            0.0,
                            "kernel with smoothing_length {} is non zero for {}",
                            smoothing_length,
                            r
                        );
                    }
                });
            }
        }
    };
}
