use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cgmath::prelude::*;
use sphrs2d::sph::smoothing_kernel::*;
use sphrs2d::units::*;

fn bench_kernels(c: &mut Criterion) {
    let smoothing_length = black_box(1.0);

    let ri_to_rj = black_box(Vector::new(1.0, 1.0) - Vector::new(0.5, 1.0));
    let r_sq = black_box(ri_to_rj.magnitude2());
    let r = black_box(r_sq.sqrt());

    {
        let kernel = black_box(CubicSpline::new(smoothing_length));
        c.bench_function("CubicSpline.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        c.bench_function("CubicSpline.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
    }
    {
        let kernel = black_box(Poly6::new(smoothing_length));
        c.bench_function("Poly6.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        c.bench_function("Poly6.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
    }
    {
        let kernel = black_box(Spiky::new(smoothing_length));
        c.bench_function("Spiky.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        c.bench_function("Spiky.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
    }
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(core::time::Duration::new(0, 100))
        .sample_size(1000)
        .significance_level(0.1)
}

criterion_group!(
    name = smoothing_kernel;
    config = config();
    targets = bench_kernels
);
criterion_main!(smoothing_kernel);
