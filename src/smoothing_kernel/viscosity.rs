use super::kernel::Kernel;
use crate::units::Direction;
//use ggez::nalgebra as na;

/// Viscosity smoothing kernel.
///
/// Müller et al.'s viscosity kernel ("Particle-Based Fluid Simulation for Interactive Applications")
/// has pretty bad properties in 2D.
/// Instead, we use a Kernel proposed by Kalle Sjöström in his Master Thesis "Computational Fluid Dynamicsin 2D Game Environments"
/// (https://pdfs.semanticscholar.org/3e9c/8e0e56d4e50da62f72002a7ad3b51b742327.pdf)
pub struct Viscosity {
    h: f32,
    hsq: f32,
    normalizer: f32,
    normalizer_laplacian: f32,
}

impl Viscosity {
    pub fn new(smoothing_length: f32) -> Viscosity {
        Viscosity {
            h: smoothing_length,
            hsq: smoothing_length * smoothing_length,
            normalizer: 90.0 / (29.0 * std::f32::consts::PI * smoothing_length * smoothing_length),
            normalizer_laplacian: 360.0 / (29.0 * std::f32::consts::PI * smoothing_length.powi(5)),
        }
    }
}

impl Kernel for Viscosity {
    #[inline]
    fn evaluate(&self, r_sq: f32) -> f32 {
        if r_sq <= self.hsq {
            let r = r_sq.sqrt(); //if r_sq < 0.00000001 { r_sq } else { r_sq.sqrt() };
            self.normalizer * (4.0 * r_sq * r / (9.0 * self.h) + r_sq) / self.hsq
        } else {
            0.0
        }
    }

    #[inline]
    fn gradient(&self, _ri_rj: Direction, _r_sq: f32) -> Direction {
        unimplemented!();
    }

    #[inline]
    fn laplacian(&self, r_sq: f32) -> f32 {
        if r_sq <= self.hsq {
            let r = if r_sq < 0.00000001 { r_sq } else { r_sq.sqrt() };
            self.normalizer_laplacian * (self.h - r)
        } else {
            0.0
        }
    }
}
