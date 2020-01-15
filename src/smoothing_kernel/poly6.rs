use super::kernel::Kernel;

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
    fn evaluate(&self, distance_sq: f32) -> f32 {
        if distance_sq <= self.hsq {
            let dsq = self.hsq - distance_sq;
            self.normalizer * dsq * dsq * dsq
        } else {
            0.0
        }
    }

    fn gradient(&self, distance_sq: f32) -> f32 {
        if distance_sq <= self.hsq {
            let dsq = self.hsq - distance_sq;
            self.normalizer_grad * dsq * dsq * distance_sq.sqrt()
        } else {
            0.0
        }
    }

    fn laplacian(&self, _distance_sq: f32) -> f32 {
        unimplemented!();
    }
}
