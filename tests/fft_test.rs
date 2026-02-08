use eigen_rs::unsupported::fft::FFT;
use eigen_rs::core::complex::Complex;

#[test]
fn test_fft_forward_inverse() {
    let mut fft = FFT::<f64>::new();
    
    // Simple signal: DC + Nyquist
    // x = [1, -1, 1, -1]
    let input = vec![
        Complex::new(1.0, 0.0),
        Complex::new(-1.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(-1.0, 0.0),
    ];
    
    let forward = fft.forward(&input);
    
    // Expected FFT: [0, 0, 4, 0] ? No.
    // DC component: sum = 0.
    // Nyquist component: alternating sum = 4.
    // Let's check magnitude.
    
    let inverse = fft.inverse(&forward);
    
    // Check round trip (unscaled)
    // inverse(forward(x)) = N * x
    let n = input.len() as f64;
    for (i, val) in inverse.iter().enumerate() {
        let expected = input[i] * Complex::new(n, 0.0);
        assert!((val.re - expected.re).abs() < 1e-10);
        assert!((val.im - expected.im).abs() < 1e-10);
    }
}

#[test]
fn test_fft_scaled_inverse() {
    let mut fft = FFT::<f64>::new();
    let input = vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ];
    
    let forward = fft.forward(&input);
    let result = fft.inverse_scaled(&forward);
    
    for (i, val) in result.iter().enumerate() {
        assert!((val.re - input[i].re).abs() < 1e-10);
        assert!((val.im - input[i].im).abs() < 1e-10);
    }
}

use eigen_rs::core::matrix::MatrixX;
use eigen_rs::unsupported::fft::FftExtension;

#[test]
fn test_fft_2d() {
    // 2x2 Matrix
    // [1, 2]
    // [3, 4]
    let mut mat = MatrixX::<Complex<f64>>::new_dynamic(2, 2).unwrap();
    *mat.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *mat.get_mut(0, 1).unwrap() = Complex::new(2.0, 0.0);
    *mat.get_mut(1, 0).unwrap() = Complex::new(3.0, 0.0);
    *mat.get_mut(1, 1).unwrap() = Complex::new(4.0, 0.0);

    let fft2_result = mat.fft2();
    let ifft2_result = fft2_result.ifft2();

    // Check round trip
    for i in 0..2 {
        for j in 0..2 {
            let original = mat.get(i, j).unwrap();
            let recovered = ifft2_result.get(i, j).unwrap();
            assert!((original.re - recovered.re).abs() < 1e-10, "Mismatch at ({},{}): Orig={:?}, Recov={:?}", i, j, original, recovered);
            assert!((original.im - recovered.im).abs() < 1e-10, "Mismatch at ({},{}): Orig={:?}, Recov={:?}", i, j, original, recovered);
        }
    }
}
