use crate::units::{Real, Vector};

/// SPH smoothing kernel
///
/// Only radially symmetric kernels are supported.
/// Assume support only within smoothing length, i.e. for |r|>h user should assume 0 as result.
/// To allow some optimizations, the actual result may be different so user needs to check!
pub trait Kernel {
    const DIVISION_EPSILON: Real = 1.0e-10;

    /// Evaluates the kernel function for a given square of distance r_sq
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn evaluate(&self, r_sq: Real, r: Real) -> Real;

    /// Evaluates the gradient of the kernel, i.e. the first derivative for a given distance r/r_sq
    /// `ri_to_rj`: Vector from a position i to a position j, so rj - ri. Not normalized!
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn gradient(&self, ri_to_rj: Vector, r_sq: Real, r: Real) -> Vector;

    /// Evaluates the laplacian of the kernel, i.e. the second derivative.
    /// `r_sq`:     Squared length of ri_to_rj
    /// `r`:        Length of ri_to_rj
    fn laplacian(&self, r_sq: Real, r: Real) -> Real;
}
