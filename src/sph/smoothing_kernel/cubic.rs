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
}

impl CubicSpline {
    pub fn new(smoothing_length: Real) -> CubicSpline {
        CubicSpline {
            h_inv: 1.0 / smoothing_length,
            normalizer: 40.0 / (7.0 * std::f64::consts::PI as Real * smoothing_length * smoothing_length),
        }
    }
}

impl Kernel for CubicSpline {
    #[inline]
    fn evaluate(&self, _r_sq: Real, r: Real) -> Real {
        let q = r * self.h_inv;
        if q <= 0.5 {
            self.normalizer * (1.0 + (q * q * q - q * q) * 6.0)
        } else if q <= 1.0 {
            self.normalizer * (1.0 - q).powi(3) * 2.0
        } else {
            0.0
        }
    }

    #[inline]
    fn gradient(&self, ri_to_rj: Vector, _r_sq: Real, r: Real) -> Vector {
        let q = r * self.h_inv;

        // todo?
        if r < 1.0e-6 {
            return cgmath::Zero::zero();
        }

        // todo optimize
        if q <= 0.5 {
            -self.normalizer * self.h_inv * q * (q * 3.0 - 2.0) * 6.0 / r * ri_to_rj
        } else if q < 1.0 {
            let factor = 1.0 - q;
            self.normalizer * self.h_inv * factor * factor * 6.0 / r * ri_to_rj
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
