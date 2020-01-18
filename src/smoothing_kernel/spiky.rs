use super::kernel::Kernel;
use crate::units::Direction;
use ggez::nalgebra as na;

/// Debrun's "Spiky" smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
/// Kernel well suited for pressure since its gradient doesn't vanish at the center.
pub struct Spiky {
    h: f32,
    hsq: f32,
    normalizer: f32,
    normalizer_grad: f32,
}

impl Spiky {
    pub fn new(smoothing_length: f32) -> Spiky {
        Spiky {
            h: smoothing_length,
            hsq: smoothing_length * smoothing_length,
            // 2D normalization factor from Salva https://github.com/rustsim/salva/blob/master/src/kernel/spiky_kernel.rs#L14
            normalizer: 10.0 / (std::f32::consts::PI * smoothing_length.powi(5)),
            normalizer_grad: -30.0 / (std::f32::consts::PI * smoothing_length.powi(5)),
        }
    }
}

impl Kernel for Spiky {
    #[inline]
    fn evaluate(&self, r_sq: f32) -> f32 {
        if r_sq <= self.hsq {
            let r = if r_sq < 0.0000001 { 0.0000001 } else { r_sq.sqrt() };
            let dsq = self.h - r;
            self.normalizer * dsq * dsq * dsq
        } else {
            0.0
        }
    }

    #[inline]
    fn gradient(&self, ri_rj: Direction, r_sq: f32) -> Direction {
        if r_sq <= self.hsq {
            let r = if r_sq < 0.0000001 { 0.0000001 } else { r_sq.sqrt() };
            let hsubr = self.h - r;
            (self.normalizer_grad * hsubr * hsubr / r) * ri_rj
        } else {
            na::zero()
        }
    }

    #[inline]
    fn laplacian(&self, _r_sq: f32) -> f32 {
        unimplemented!();
    }
}
