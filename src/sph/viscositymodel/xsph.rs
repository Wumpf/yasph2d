use super::ViscosityModel;

use super::super::smoothing_kernel::*;
use crate::units::*;

// XSPH as in "Ghost SPH for Animating Water", Schechter et al. (https://www.cs.ubc.ca/~rbridson/docs/schechter-siggraph2012-ghostsph.pdf)
pub struct XSPHViscosityModel {
    pub epsilon: Real, // default 0.05
    kernel: Poly6,
}
impl XSPHViscosityModel {
    pub fn new(smoothing_length: Real) -> XSPHViscosityModel {
        XSPHViscosityModel {
            epsilon: 0.05,
            kernel: Poly6::new(smoothing_length),
        }
    }
}
impl ViscosityModel for XSPHViscosityModel {
    #[inline]
    fn compute_viscous_accelleration(&self, dt: Real, r_sq: Real, r: Real, massj: Real, rhoj: Real, velocitydiff: Vector) -> Vector {
        self.epsilon * massj * self.kernel.evaluate(r_sq, r) / (rhoj * dt) * velocitydiff
    }
}
