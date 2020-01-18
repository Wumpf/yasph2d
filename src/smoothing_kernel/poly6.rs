use super::kernel::Kernel;
use crate::units::Direction;

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
            normalizer: 4.0 / (std::f32::consts::PI * smoothing_length.powi(8)),
            normalizer_grad: -24.0 / (std::f32::consts::PI * smoothing_length.powi(8)),
        }
    }
}

impl Kernel for Poly6 {
    #[inline]
    fn evaluate(&self, r_sq: f32) -> f32 {
        let dsq = self.hsq - r_sq;
        self.normalizer * dsq * dsq * dsq
    }

    #[inline]
    fn gradient(&self, ri_rj: Direction, r_sq: f32) -> Direction {
        let hsq_sub_rsq = self.hsq - r_sq;
        self.normalizer_grad * hsq_sub_rsq * hsq_sub_rsq * ri_rj
    }

    #[inline]
    fn laplacian(&self, _r_sq: f32) -> f32 {
        unimplemented!();
    }
}
