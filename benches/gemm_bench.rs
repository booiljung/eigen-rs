use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eigen_rs::core::matrix::MatrixX;

fn bench_gemm_large(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];

    let mut group = c.benchmark_group("GEMM Large");
    group.sample_size(10); // Reduce sample size for large matrices

    for size in sizes {
        // F32
        {
            let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
            let mut b = MatrixX::<f32>::new_dynamic(size, size).unwrap();
            let mut res = MatrixX::<f32>::new_dynamic(size, size).unwrap();
            a.set_constant(1.0);
            b.set_constant(1.0);

            // Optimized
            group.throughput(Throughput::Elements((size * size * size) as u64));
            group.bench_with_input(BenchmarkId::new("f32_opt", size), &size, |bencher, _| {
                bencher.iter(|| {
                    res.assign_product(&(black_box(&a) * black_box(&b)))
                        .unwrap();
                });
            });

            // Baseline
            group.bench_with_input(BenchmarkId::new("f32_base", size), &size, |bencher, _| {
                bencher.iter(|| {
                    eigen_rs::core::ops::gemm::gemm_cm_unoptimized_xpr(&a, &b, &mut res).unwrap();
                });
            });
        }

        // F64
        {
            let mut a = MatrixX::<f64>::new_dynamic(size, size).unwrap();
            let mut b = MatrixX::<f64>::new_dynamic(size, size).unwrap();
            let mut res = MatrixX::<f64>::new_dynamic(size, size).unwrap();
            a.set_constant(1.0);
            b.set_constant(1.0);

            // Optimized
            group.throughput(Throughput::Elements((size * size * size) as u64));
            group.bench_with_input(BenchmarkId::new("f64_opt", size), &size, |bencher, _| {
                bencher.iter(|| {
                    res.assign_product(&(black_box(&a) * black_box(&b)))
                        .unwrap();
                });
            });

            // Baseline
            group.bench_with_input(BenchmarkId::new("f64_base", size), &size, |bencher, _| {
                bencher.iter(|| {
                    eigen_rs::core::ops::gemm::gemm_cm_unoptimized_xpr(&a, &b, &mut res).unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_gemm_large);
criterion_main!(benches);
