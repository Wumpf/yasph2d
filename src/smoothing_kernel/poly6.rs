use super::kernel::Kernel;
use crate::units::{Real, Vector};

/// Poly6 smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
/// Kernel not well suited for computing pressure forces since derivative approaches zero.
pub struct Poly6 {
    hsq: f32,
    normalizer: f32,
    normalizer_grad: f32,
}

impl Poly6 {
    pub fn new(smoothing_length: f32) -> Poly6 {
        Poly6 {
            hsq: smoothing_length * smoothing_length,
            // 2D normalization factor from Salva https://github.com/rustsim/salva/blob/master/src/kernel/poly6_kernel.rs#L14
            normalizer: 4.0 / (std::f64::consts::PI as Real * smoothing_length.powi(8)),
            normalizer_grad: -24.0 / (std::f64::consts::PI as Real * smoothing_length.powi(8)),
        }
    }
}

impl Kernel for Poly6 {
    #[inline]
    fn evaluate(&self, r_sq: Real, _r: Real) -> Real {
        let dsq = self.hsq - r_sq;
        self.normalizer * dsq * dsq * dsq
    }

    #[inline]
    fn gradient(&self, ri_to_rj: Vector, r_sq: Real, _r: Real) -> Vector {
        let hsq_sub_rsq = self.hsq - r_sq;
        self.normalizer_grad * hsq_sub_rsq * hsq_sub_rsq * ri_to_rj
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        unimplemented!();
    }
}
