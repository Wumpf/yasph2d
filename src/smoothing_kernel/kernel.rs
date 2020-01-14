/// SPH smoothing kernel
pub trait Kernel {
    /// Evaluates the kernel function.
    fn evaluate(&self, distance_sq: f32) -> f32;

    /// Evaluates the gradient of the kernel, i.e. the first derivative.
    fn gradient(&self, distance_sq: f32) -> f32;

    /// Evaluates the laplacian of the kernel, i.e. the second derivative.
    fn laplacian(&self, distance_sq: f32) -> f32;
}
