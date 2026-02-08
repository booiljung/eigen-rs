use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eigen_rs::core::matrix::MatrixX;

fn bench_matrix_multiplication(c: &mut Criterion) {
    let sizes = [64, 128, 256];

    let mut group = c.benchmark_group("Matrix Multiplication");
    for size in sizes {
        let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
        let mut b = MatrixX::<f32>::new_dynamic(size, size).unwrap();
        let mut res = MatrixX::<f32>::new_dynamic(size, size).unwrap();

        // Fill with some data
        for i in 0..(size * size) {
            let row = i % size;
            let col = i / size;
            *a.get_mut(row, col).unwrap() = i as f32;
            *b.get_mut(row, col).unwrap() = (i * 2) as f32;
        }

        group.bench_with_input(
            criterion::BenchmarkId::new("f32", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    res.assign_product(&(black_box(&a) * black_box(&b)))
                        .unwrap();
                });
            },
        );

        let mut a_d = MatrixX::<f64>::new_dynamic(size, size).unwrap();
        let mut b_d = MatrixX::<f64>::new_dynamic(size, size).unwrap();
        let mut res_d = MatrixX::<f64>::new_dynamic(size, size).unwrap();
        for i in 0..(size * size) {
            *a_d.get_mut(i % size, i / size).unwrap() = i as f64;
            *b_d.get_mut(i % size, i / size).unwrap() = (i * 2) as f64;
        }

        group.bench_with_input(
            criterion::BenchmarkId::new("f64", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    res_d
                        .assign_product(&(black_box(&a_d) * black_box(&b_d)))
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_cholesky_llt(c: &mut Criterion) {
    let sizes = [64, 128, 256];
    let mut group = c.benchmark_group("LLT Decomposition");
    for size in sizes {
        // Create a symmetric positive definite matrix
        let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
        for i in 0..size {
            for j in 0..size {
                let val = if i == j { size as f32 * 2.0 } else { 1.0 };
                *a.get_mut(i, j).unwrap() = val;
            }
        }

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            &size,
            |bencher, &s| {
                bencher.iter(|| {
                    black_box(&a).llt().unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_svd_jacobi(c: &mut Criterion) {
    let sizes = [32, 64]; // SVD is slow, keep sizes small for now
    let mut group = c.benchmark_group("Jacobi SVD");
    for size in sizes {
        let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
        for i in 0..size {
            for j in 0..size {
                *a.get_mut(i, j).unwrap() = (i + j) as f32;
            }
        }
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(&a).jacobi_svd().unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_eigen_selfadjoint(c: &mut Criterion) {
    let sizes = [32, 64];
    let mut group = c.benchmark_group("SelfAdjoint EigenSolver");
    for size in sizes {
        let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
        // Make it symmetric
        for i in 0..size {
            for j in 0..size {
                *a.get_mut(i, j).unwrap() = (i + j) as f32;
            }
        }
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(&a).self_adjoint_eigen_solver(true).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_multiplication,
    bench_cholesky_llt,
    bench_svd_jacobi,
    bench_eigen_selfadjoint
);
criterion_main!(benches);
