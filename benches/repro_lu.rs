use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use eigen_rs::core::decompositions::lu::PartialPivLU;
use eigen_rs::core::matrix::Matrix;

fn benchmark_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("LU_Optimization");
    group.sample_size(10); // Low sample size for speed during dev

    for size in [4, 8, 12, 16, 24, 32].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Generate random matrix
        let mut mat =
            Matrix::<f64, eigen_rs::core::storage::DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        // Simple LCG for deterministic "random" numbers
        let mut state: u64 = 42;
        for i in 0..n {
            for j in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = (state >> 33) as f64 / 2147483648.0;
                *mat.get_mut(i, j).unwrap() = val;
            }
        }
        group.bench_function(format!("PartialPivLU_N{}", n), |b| {
            b.iter(|| {
                let res = PartialPivLU::new(black_box(&mat));
                black_box(res);
            })
        });

        // Also bench MatMul to see GEMM overhead
        let mut mat_b =
            Matrix::<f64, eigen_rs::core::storage::DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        // fill b
        for i in 0..n {
            for j in 0..n {
                *mat_b.get_mut(i, j).unwrap() = 1.0;
            }
        }

        group.bench_function(format!("MatMul_N{}", n), |b| {
            b.iter(|| {
                let mut res =
                    Matrix::<f64, eigen_rs::core::storage::DynamicStorage<f64>>::new_dynamic(n, n)
                        .unwrap();
                res.assign(&(&mat * &mat_b)).unwrap();
                black_box(res);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_lu);
criterion_main!(benches);
