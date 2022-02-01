use super::kernel::Kernel;
use crate::units::{Real, Vector};

// Impl / normalization factors via
// https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/d5172c9/SPlisHSPlasH/SPHKernels.h#L545
//
// An interesting property of WendlandQuintic described in "Improving convergence in smoothed particle hydrodynamics simulations without pairing instability" by Walter Dehnen and Hossam Aly:
// https://arxiv.org/pdf/1204.2471.pdf
// "Linear stability analysis in three dimensions and test simulations demonstrate that the Wend-
// land kernels avoid the pairing instability for all NH, despite having vanishing derivative at the
// origin"
//
// Evaluating this Kernel is also quite a bit faster than CubicSpline.
// btw. Poly6 is fastest but demonstrates strong pairing instability when tested in this project
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
            normalizer: 4.0 * 7.0 / (std::f64::consts::PI as Real * smoothing_length.powi(2)),
            normalizer_grad: 140.0 / (std::f64::consts::PI as Real * smoothing_length.powi(4)),
        }
    }
}

impl Kernel for WendlandQuinticC2 {
    #[inline]
    fn evaluate(&self, _r_sq: Real, r: Real) -> Real {
        let q = (self.h_inv * r).min(1.0);
        let one_minus_q = 1.0 - q;
        let one_minus_q_sq = one_minus_q * one_minus_q;
        self.normalizer * one_minus_q_sq * one_minus_q_sq * (q + 0.25)
    }

    #[inline]
    fn gradient(&self, ri_to_rj: Vector, _r_sq: Real, r: Real) -> Vector {
        let q = (r * self.h_inv).min(1.0);
        let one_minus_q = 1.0 - q;
        (self.normalizer_grad * one_minus_q * one_minus_q * one_minus_q) * ri_to_rj
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        unimplemented!();
    }
}

generate_kernel_tests!(WendlandQuinticC2);
