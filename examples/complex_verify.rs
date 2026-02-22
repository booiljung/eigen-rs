use eigen_rs::core::decompositions::lu::PartialPivLU;
use eigen_rs::core::matrix::Matrix;
use num_complex::Complex;

fn main() {
    println!("Verifying Complex Number Support...");

    // Create a 3x3 Complex Matrix
    let mut a = Matrix::<Complex<f64>, _>::new_dynamic(3, 3).unwrap();

    // Fill with values:
    // [ 1+i, 2-i, 3 ]
    // [ 4,   5+i, 6 ]
    // [ 7-i, 8,   9 ]
    *a.get_mut(0, 0).unwrap() = Complex::new(1.0, 1.0);
    *a.get_mut(0, 1).unwrap() = Complex::new(2.0, -1.0);
    *a.get_mut(0, 2).unwrap() = Complex::new(3.0, 0.0);

    *a.get_mut(1, 0).unwrap() = Complex::new(4.0, 0.0);
    *a.get_mut(1, 1).unwrap() = Complex::new(5.0, 1.0);
    *a.get_mut(1, 2).unwrap() = Complex::new(6.0, 0.0);

    *a.get_mut(2, 0).unwrap() = Complex::new(7.0, -1.0);
    *a.get_mut(2, 1).unwrap() = Complex::new(8.0, 0.0);
    *a.get_mut(2, 2).unwrap() = Complex::new(9.0, 0.0);

    println!("Matrix A:\n{:?}", a);

    // Compute LU
    let lu = PartialPivLU::new(&a).unwrap();
    println!("LU Decomposition computed.");

    // Verify A * x = b
    let mut b = Matrix::<Complex<f64>, _>::new_dynamic(3, 1).unwrap();
    *b.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *b.get_mut(1, 0).unwrap() = Complex::new(2.0, 0.0);
    *b.get_mut(2, 0).unwrap() = Complex::new(3.0, 0.0);

    let x = lu.solve(&b).unwrap();
    println!("Solution x:\n{:?}", x);

    // Check residual: r = ||Ax - b||
    let mut ax = Matrix::<Complex<f64>, _>::new_dynamic(3, 1).unwrap();
    ax.assign(&(&a * &x)).unwrap();

    let mut r = Matrix::<Complex<f64>, _>::new_dynamic(3, 1).unwrap();
    r.assign(&(&ax - &b)).unwrap();
    let r_norm = r.norm();

    println!("Residual norm: {}", r_norm);

    if r_norm < 1e-12 {
        println!("SUCCESS: Residual is small.");
    } else {
        println!("FAILURE: Residual is too large.");
        std::process::exit(1);
    }
}
