use criterion::{black_box, criterion_group, Criterion};

use sphrs2d::sph::morton::*;

fn bench_morton(c: &mut Criterion) {
    let x = 123;
    let y = 321;

    let mut group = c.benchmark_group("morton.encode");
    group.bench_function("encode_bitfiddle", |b| b.iter(|| encode_bitfiddle(black_box(x), black_box(y))));
    group.bench_function("encode_lookup", |b| b.iter(|| encode_lookup(black_box(x), black_box(y))));
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
}

criterion_group!(
    name = morton;
    config = config();
    targets = bench_morton
);
