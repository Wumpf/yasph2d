use super::kernel::Kernel;
use crate::units::{Real, Vector};

/// Cubic Spline smoothing kernel.
///
/// Classic cubic spline cernel from "J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574."
/// Normalization factors from https://pysph.readthedocs.io/en/latest/reference/kernels.html#monaghan1992
#[derive(Copy, Clone)]
pub struct CubicSpline {
    h_inv: Real,
    normalizer: Real,
    normalizer_grad: Real,
}

impl CubicSpline {
    pub fn new(smoothing_length: Real) -> CubicSpline {
        CubicSpline {
            h_inv: 1.0 / smoothing_length,
            normalizer: 6.0 * 40.0 / (7.0 * std::f64::consts::PI as Real * smoothing_length * smoothing_length),
            normalizer_grad: 6.0 * 40.0 / (7.0 * std::f64::consts::PI as Real * smoothing_length * smoothing_length * smoothing_length),
        }
    }
}

impl Kernel for CubicSpline {
    #[inline]
    fn evaluate(&self, _r_sq: Real, r: Real) -> Real {
        let q = r * self.h_inv;
        if q <= 0.5 {
            let q_sq = q * q;
            self.normalizer * ((1.0/6.0) + q_sq * q - q_sq)
        } else if q <= 1.0 {
            let one_minus_q = 1.0 - q;
            self.normalizer * one_minus_q * one_minus_q * one_minus_q * (2.0 / 6.0)
        } else {
            0.0
        }
    }

    #[inline]
    fn gradient(&self, ri_to_rj: Vector, _r_sq: Real, r: Real) -> Vector {
        let q = r * self.h_inv;
        if q <= 0.5 {
            self.normalizer_grad * q * (2.0 - q * 3.0) / r * ri_to_rj
        } else if q < 1.0 {
            let factor = 1.0 - q;
            self.normalizer_grad * factor * factor / r * ri_to_rj
        } else {
            cgmath::Zero::zero()
        }
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        unimplemented!();
    }
}

generate_kernel_tests!(CubicSpline);
