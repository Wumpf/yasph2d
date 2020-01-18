use super::kernel::Kernel;
use crate::units::{Direction, Real};

/// Debrun's "Spiky" smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
/// Kernel well suited for pressure since its gradient doesn't vanish at the center.
pub struct Spiky {
    h: Real,
    normalizer: Real,
    normalizer_grad: Real,
}

impl Spiky {
    pub fn new(smoothing_length: Real) -> Spiky {
        Spiky {
            h: smoothing_length,
            // 2D normalization factor from Salva https://github.com/rustsim/salva/blob/master/src/kernel/spiky_kernel.rs#L14
            normalizer: 10.0 / (std::f64::consts::PI as Real * smoothing_length.powi(5)),
            normalizer_grad: -30.0 / (std::f64::consts::PI as Real * smoothing_length.powi(5)),
        }
    }
}

impl Kernel for Spiky {
    #[inline]
    fn evaluate(&self, _r_sq: Real, r: Real) -> Real {
        let hsubr = self.h - r;
        self.normalizer * hsubr * hsubr * hsubr
    }

    #[inline]
    fn gradient(&self, ri_to_rj: Direction, _r_sq: Real, r: Real) -> Direction {
        let hsubr = self.h - r;
        (self.normalizer_grad * hsubr * hsubr / (r + Self::DIVISION_EPSILON)) * ri_to_rj
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        unimplemented!();
    }
}
