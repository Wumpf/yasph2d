use super::viscositymodel::ViscosityModel;

use crate::units::*;
use super::super::smoothing_kernel::*;

// Laplacian based physical model as in "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
pub struct PhysicalViscosityModel {
    pub fluid_viscosity: Real, // the dynamic viscosity of this fluid in Pa*s (μ, mu)
    kernel: Viscosity,
}
impl PhysicalViscosityModel {
    pub fn new(smoothing_length: Real) -> PhysicalViscosityModel {
        PhysicalViscosityModel {
            fluid_viscosity: 1.0016 / 1000.0, // Water is 1.0016 / 1000.0, // viscosity of water at 20 degrees in Pa*s
            kernel: Viscosity::new(smoothing_length),
        }
    }
}
impl ViscosityModel for PhysicalViscosityModel {
    #[inline]
    fn compute_viscous_accelleration(&self, _dt: Real, r_sq: Real, r: Real, massj: Real, rhoj: Real, velocitydiff: Vector) -> Vector {
        self.fluid_viscosity * massj * self.kernel.laplacian(r_sq, r) / rhoj * velocitydiff
    }
}