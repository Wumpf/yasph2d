use criterion::{black_box, criterion_group, Criterion};

use cgmath::prelude::*;
use yasph2d::sph::smoothing_kernel::*;
use yasph2d::units::*;

fn bench_kernels(c: &mut Criterion) {
    let smoothing_length = black_box(1.0);

    let ri_to_rj = black_box(Vector::new(1.0, 1.0) - Vector::new(0.5, 1.0));
    let r_sq = black_box(ri_to_rj.magnitude2());
    let r = black_box(r_sq.sqrt());

    // Note:
    // Tried differing output every iteration using iter_batched, however this has still way too much overhead on the incredibly fast kernel funcs.
    // It seems Criterion is not really suited to benchmark functions _that_ small.

    {
        let mut group_eval = c.benchmark_group("smoothing_kernel.evaluate");
        {
            let kernel = black_box(CubicSpline::new(smoothing_length));
            group_eval.bench_function("CubicSpline.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        }
        {
            let kernel = black_box(Poly6::new(smoothing_length));
            group_eval.bench_function("Poly6.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        }
        {
            let kernel = black_box(Spiky::new(smoothing_length));
            group_eval.bench_function("Spiky.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        }
        {
            let kernel = black_box(WendlandQuinticC2::new(smoothing_length));
            group_eval.bench_function("WendlandQuinticC2.evaluate", |b| b.iter(|| kernel.evaluate(r_sq, r)));
        }
        group_eval.finish();
    }
    {
        let mut group_grad = c.benchmark_group("smoothing_kernel.gradient");
        {
            let kernel = black_box(CubicSpline::new(smoothing_length));
            group_grad.bench_function("CubicSpline.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
        }
        {
            let kernel = black_box(Poly6::new(smoothing_length));
            group_grad.bench_function("Poly6.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
        }
        {
            let kernel = black_box(Spiky::new(smoothing_length));
            group_grad.bench_function("Spiky.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
        }
        {
            let kernel = black_box(Spiky::new(smoothing_length));
            group_grad.bench_function("WendlandQuinticC2.gradient", |b| b.iter(|| kernel.gradient(ri_to_rj, r_sq, r)));
        }
        group_grad.finish();
    }
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(core::time::Duration::new(0, 100))
        .sample_size(1000)
        .noise_threshold(0.05)
}

criterion_group!(
    name = smoothing_kernel;
    config = config();
    targets = bench_kernels
);
