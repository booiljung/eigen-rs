use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::decompositions::lu::PartialPivLU;
use num_complex::Complex;

fn main() {
    println!("Verifying Complex LU Decomposition...");

    // A = [[1, i], [i, 1]]
    // Det = 2.
    let mut a = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    *a.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *a.get_mut(0, 1).unwrap() = Complex::new(0.0, 1.0);
    *a.get_mut(1, 0).unwrap() = Complex::new(0.0, 1.0);
    *a.get_mut(1, 1).unwrap() = Complex::new(1.0, 0.0);

    println!("Matrix A:\n{:?}", a);

    let lu = PartialPivLU::new(&a).unwrap();
    
    // Verify PA = LU
    // Or solve Ax = b
    let b = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    let mut b_ones = b.clone();
    *b_ones.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *b_ones.get_mut(1, 0).unwrap() = Complex::new(1.0, 0.0);

    // Solve
    let x = lu.solve(&b_ones).unwrap();
    println!("Solution x for b=[1, 1]:\n{:?}", x);

    // Reconstruct Ax
    let mut ax = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    ax.assign(&(&a * &x)).unwrap();
    println!("A * x:\n{:?}", ax);

    let mut diff = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    diff.assign(&(&ax - &b_ones)).unwrap();
    
    let norm = diff.norm();
    println!("Residual Norm: {:?}", norm);

    if norm < 1e-10 {
        println!("LU Verification: PASSED");
    } else {
        println!("LU Verification: FAILED");
        std::process::exit(1);
    }
}
