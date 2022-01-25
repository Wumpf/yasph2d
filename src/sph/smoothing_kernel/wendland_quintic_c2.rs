use cgmath::num_traits::Pow;

use super::kernel::Kernel;
use crate::units::{Real, Vector};

// https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/d5172c9/SPlisHSPlasH/SPHKernels.h#L545
#[derive(Copy, Clone)]
pub struct WendlandQuinticC2 {
    h_inv: Real,
    normalizer: Real,
    normalizer_grad: Real,
}

impl WendlandQuinticC2 {
    pub fn new(smoothing_length: Real) -> Self {
        WendlandQuinticC2 {
            h_inv: 1.0 / smoothing_length,
            normalizer: 7.0 / (std::f64::consts::PI as Real * smoothing_length.powi(2)),
            normalizer_grad: 140.0 / (std::f64::consts::PI as Real * smoothing_length.powi(2)),
        }
    }
}

impl Kernel for WendlandQuinticC2 {
    #[inline]
    fn evaluate(&self, _r_sq: Real, r: Real) -> Real {
        let q = (self.h_inv * r).min(1.0);
        let one_minus_q = 1.0 - q;
        let one_minus_q_sq = one_minus_q * one_minus_q;
        self.normalizer * one_minus_q_sq * one_minus_q_sq * (4.0 * q + 1.0)
    }

    #[inline]
    fn gradient(&self, ri_to_rj: Vector, _r_sq: Real, r: Real) -> Vector {
        let q = (r * self.h_inv).min(1.0);
        let gradq = ri_to_rj * (self.h_inv / r);
        (self.normalizer_grad * q * (1.0 - q).pow(3.0)) * gradq
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        unimplemented!();
    }
}

generate_kernel_tests!(WendlandQuinticC2);
