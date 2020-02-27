use crate::units::{Point, Real, Vector};
use cgmath::prelude::*;

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

    #[inline(always)]
    fn gradient_from_positions(&self, ri: Point, rj: Point) -> Vector {
        let ri_to_rj = rj - ri;
        let r_sq = ri_to_rj.magnitude2();
        let r = r_sq.sqrt();
        self.gradient(ri_to_rj, r_sq, r)
    }

    /// Evaluates the laplacian of the kernel, i.e. the second derivative.
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn laplacian(&self, r_sq: Real, r: Real) -> Real;
}

// TODO:
// * Try WendlandQuintic: https://pysph.readthedocs.io/en/latest/reference/kernels.html#pysph.base.kernels.WendlandQuintic
// * consider removing laplacian alltogether

macro_rules! generate_kernel_tests {
    ($kernel_type:ident) => {
        #[cfg(test)]
        mod tests {
            use super::*;
            use cgmath::prelude::*;

            pub static TEST_SMOOTHING_LENGTHS: [Real; 3] = [0.5, 1.0, 123.0];

            fn run_for_different_kernel_sizes(func: impl Fn($kernel_type, Real)) {
                for &smoothing_length in TEST_SMOOTHING_LENGTHS.iter() {
                    func($kernel_type::new(smoothing_length), smoothing_length);
                }
            }

            fn integrate_over_domain(smoothing_length: Real, func: impl Fn(Vector) -> Real) -> Real {
                let mut accumulator = 0.0;
                const SAMPLES_PER_AXIS: usize = 200;
                for x in 0..SAMPLES_PER_AXIS {
                    for y in 0..SAMPLES_PER_AXIS {
                        let p = Vector::new(x as Real, y as Real) / (SAMPLES_PER_AXIS - 1) as Real * smoothing_length * 2.0
                            - Vector::new(smoothing_length, smoothing_length);
                        accumulator += func(p);
                    }
                }
                accumulator *= (2.0 * smoothing_length / SAMPLES_PER_AXIS as Real).powi(2); // reactangle rule. Should probably use simpson or similar
                accumulator
            }

            fn iterate_over_domain(smoothing_length: Real, func: impl Fn(Vector)) {
                integrate_over_domain(smoothing_length, |p| {
                    func(p);
                    0.0
                });
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

            #[test]
            fn evaluate_is_always_positive() {
                run_for_different_kernel_sizes(|kernel, smoothing_length| {
                    iterate_over_domain(smoothing_length, |p| {
                        assert_ge!(kernel.evaluate(p.magnitude2(), p.magnitude()), 0.0);
                    });
                });
            }

            #[test]
            fn integrates_to_one_over_domain() {
                run_for_different_kernel_sizes(|kernel, smoothing_length| {
                    let integral = integrate_over_domain(smoothing_length, |p| kernel.evaluate(p.magnitude2(), p.magnitude()));
                    assert_lt!((1.0 - integral).abs(), 0.01);
                });
            }

            #[test]
            fn gradient_is_similar_to_numerical_gradient() {
                run_for_different_kernel_sizes(|kernel, smoothing_length| {
                    iterate_over_domain(smoothing_length, |p| {
                        let analytical_gradient = kernel.gradient(p, p.magnitude2(), p.magnitude());

                        let step = smoothing_length * 0.0001;
                        let pxpos = p + Vector::new(step, 0.0);
                        let pxneg = p - Vector::new(step, 0.0);
                        let pypos = p + Vector::new(0.0, step);
                        let pyneg = p - Vector::new(0.0, step);
                        let numerical_gradient = Vector::new(
                            kernel.evaluate(pxneg.magnitude2(), pxneg.magnitude()) - kernel.evaluate(pxpos.magnitude2(), pxpos.magnitude()),
                            kernel.evaluate(pyneg.magnitude2(), pyneg.magnitude()) - kernel.evaluate(pypos.magnitude2(), pypos.magnitude()),
                        ) / step
                            * 0.5;

                        const RELATIVE_ERROR_EPS: Real = 0.00001;
                        assert_lt!(
                            (1.0 - (numerical_gradient.magnitude() + RELATIVE_ERROR_EPS) / (analytical_gradient.magnitude() + RELATIVE_ERROR_EPS))
                                .abs(),
                            0.05,
                            "relative magnitude error of gradient too high - analytical_gradient {:?}, numerical_gradient {:?}",
                            analytical_gradient,
                            numerical_gradient
                        );
                        let dotproduct = numerical_gradient.dot(analytical_gradient) + RELATIVE_ERROR_EPS;
                        assert_lt!(
                            (dotproduct / (analytical_gradient.magnitude2() + RELATIVE_ERROR_EPS) - 1.0).abs(),
                            0.05,
                            "direction error of gradient too high - analytical_gradient {:?}, numerical_gradient {:?}",
                            analytical_gradient,
                            numerical_gradient
                        );
                    });
                });
            }
        }
    };
}
