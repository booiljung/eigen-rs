use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eigen_rs::unsupported::splines::Spline;
use eigen_rs::core::storage::Storage;

fn spline_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spline");

    // Spline Interpolation setup
    let points_vec = vec![
        0.0, 0.0, 
        1.0, 2.0, 
        2.0, 1.0, 
        3.0, 3.0, 
        4.0, 2.0
    ];
    let mut points = eigen_rs::core::matrix::MatrixX::<f64>::new_dynamic(2, 5).unwrap();
    points.storage_mut().data_mut().copy_from_slice(&points_vec);

    group.bench_function("Interpolate 5 points", |b| {
        b.iter(|| {
            Spline::interpolate(black_box(&points), 3)
        })
    });

    let spline = Spline::interpolate(&points, 3).unwrap();
    
    group.bench_function("Eval Spline", |b| {
        b.iter(|| {
            spline.eval(black_box(1.5))
        })
    });

    group.finish();
}

criterion_group!(benches, spline_benchmark);
criterion_main!(benches);
