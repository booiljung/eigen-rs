use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eigen_rs::core::complex::Complex;
use eigen_rs::core::matrix::MatrixX;
use eigen_rs::unsupported::fft::FftExtension;
use eigen_rs::unsupported::fft::FFT;

fn fft_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");

    // 1D FFT Benchmark
    let size = 1024;
    let input: Vec<Complex<f64>> = (0..size).map(|i| Complex::new(i as f64, 0.0)).collect();
    let mut fft = FFT::new();

    group.bench_function("FFT 1D 1024", |b| b.iter(|| fft.forward(black_box(&input))));

    // 2D FFT Benchmark (256x256)
    let rows = 256;
    let cols = 256;
    let mut mat = MatrixX::<Complex<f64>>::new_dynamic(rows, cols).unwrap();
    // Fill
    for i in 0..rows {
        for j in 0..cols {
            *mat.get_mut(i, j).unwrap() = Complex::new((i + j) as f64, 0.0);
        }
    }

    group.bench_function("FFT 2D 256x256", |b| b.iter(|| mat.fft2()));

    group.finish();
}

criterion_group!(benches, fft_benchmark);
criterion_main!(benches);
