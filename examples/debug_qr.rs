use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::decompositions::qr::HouseholderQR;
use num_complex::Complex;
use eigen_rs::core::storage::DynamicStorage;

fn main() {
    println!("Verifying Complex QR...");

    // A = [[1, i], [i, 1]]
    let mut a = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    *a.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *a.get_mut(0, 1).unwrap() = Complex::new(0.0, 1.0);
    *a.get_mut(1, 0).unwrap() = Complex::new(0.0, 1.0);
    *a.get_mut(1, 1).unwrap() = Complex::new(1.0, 0.0);

    println!("Matrix A:\n{:?}", a);

    // Inspect QR internals
    println!("(Debug print from library should appear below if added)");
    
    let qr = HouseholderQR::new(&a).unwrap();
    let b = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    let mut b_ones = b.clone();
    *b_ones.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *b_ones.get_mut(1, 0).unwrap() = Complex::new(1.0, 0.0);

    let x = qr.solve(&b_ones).unwrap();
    println!("Solution x for b=[1, 1]:");
    for i in 0..2 {
        println!("  Row {}: {}", i, *x.get(i, 0).unwrap());
    }

    let mut ax_mat = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    ax_mat.assign(&(&a * &x)).unwrap();
    println!("A * x:");
    for i in 0..2 {
        println!("  Row {}: {}", i, *ax_mat.get(i, 0).unwrap());
    }
    
    // Manual check: A x = b
    // [[1, i], [i, 1]] [x0, x1] = [1, 1]
    // x0 + i x1 = 1
    // i x0 + x1 = 1 => x1 = 1 - i x0
    // x0 + i(1 - i x0) = 1
    // x0 + i + x0 = 1 => 2 x0 = 1 - i => x0 = 0.5 - 0.5i
    // x1 = 1 - i(0.5 - 0.5i) = 1 - 0.5i - 0.5 = 0.5 - 0.5i
    // So target x is [0.5-0.5i, 0.5-0.5i].
    
    let target_x0 = Complex::new(0.5, -0.5);
    println!("Target x: {}, {}", target_x0, target_x0);
    
    // Or just iterate
    println!("Residual components check:");
    for i in 0..2 {
        let d = *ax_mat.get(i, 0).unwrap() - *b_ones.get(i, 0).unwrap();
        println!("  Row {}: {}", i, d);
    }
}
