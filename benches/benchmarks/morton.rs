use criterion::{black_box, criterion_group, Criterion};

use yasph2d::sph::morton::*;

fn bench_morton(c: &mut Criterion) {
    {
        let min = encode(2, 2);
        let max = encode(3, 6);
        let cur = encode(4, 0);

        // TODO: How to construct a bad case for this? perf is highly input dependent
        c.bench_function("find_bigmin", |b| b.iter(|| find_bigmin(black_box(cur), black_box(min), black_box(max))));
    }
    {
        let x: u32 = 123;
        let y: u32 = 321;

        let mut group = c.benchmark_group("morton.encode");
        group.bench_function("encode_bitfiddle", |b| {
            b.iter(|| encode_bitfiddle(black_box(x) as u16, black_box(y) as u16))
        });
        group.bench_function("encode_lookup", |b| b.iter(|| encode_lookup(black_box(x) as u16, black_box(y) as u16)));
        group.finish();
    }
}

fn config() -> Criterion {
    Criterion::default().warm_up_time(core::time::Duration::new(0, 500)).noise_threshold(0.05)
}

criterion_group!(
    name = morton;
    config = config();
    targets = bench_morton
);
