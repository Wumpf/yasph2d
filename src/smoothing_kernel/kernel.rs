use crate::units::Direction;

/// SPH smoothing kernel
///
/// Only radially symmetric kernels are supported.
/// Assume support only within smoothing length, i.e. for |r|>h user should assume 0 as result.
/// To allow some optimizations, the actual result may be different so user needs to check!
pub trait Kernel {
    /// Evaluates the kernel function for a given square of distance r_sq
    fn evaluate(&self, r_sq: f32) -> f32;

    /// Evaluates the gradient of the kernel, i.e. the first derivative for a given distance r/r_sq
    /// `ri_rj`:    Direction from a position j to a position i
    /// `r_sq`:     Squared length of ri_rj
    fn gradient(&self, ri_rj: Direction, r_sq: f32) -> Direction;

    /// Evaluates the laplacian of the kernel, i.e. the second derivative.
    fn laplacian(&self, r_sq: f32) -> f32;
}
