use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
use eigen_rs::core::decompositions::llt::LLT;
use eigen_rs::core::decompositions::ldlt::LDLT;
use eigen_rs::core::decompositions::qr::HouseholderQR;
use eigen_rs::core::scalar::Scalar;
use num_complex::Complex;
use num_traits::{Zero, One};

fn main() {
    println!("Verifying Complex Cholesky and QR Decompositions...");
    
    verify_llt();
    verify_ldlt();
    verify_qr();
}

fn adjoint<T: Scalar>(mat: &Matrix<T, DynamicStorage<T>>) -> Matrix<T, DynamicStorage<T>> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut res = Matrix::<T, DynamicStorage<T>>::new_dynamic(cols, rows).unwrap();
    for i in 0..rows {
        for j in 0..cols {
             *res.get_mut(j, i).unwrap() = mat.get(i, j).unwrap().conj();
        }
    }
    res
}

fn verify_llt() {
    println!("\n--- Verifying LLT (L * L^H) ---");
    // Create Hermitian Positive Definite Matrix
    // A = [ 4  2i ]
    //     [-2i 5  ]
    let mut a = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    *a.get_mut(0, 0).unwrap() = Complex::new(4.0, 0.0);
    *a.get_mut(0, 1).unwrap() = Complex::new(0.0, 2.0);
    *a.get_mut(1, 0).unwrap() = Complex::new(0.0, -2.0);
    *a.get_mut(1, 1).unwrap() = Complex::new(5.0, 0.0);

    println!("Matrix A:\n{:?}", a);

    let llt = a.llt().unwrap();
    let l = llt.matrix_l();
    println!("Matrix L:\n{:?}", l);

    // Reconstruct A = L * L^H
    let l_h = adjoint(l);
    let mut recon = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    recon.assign(&(l * &l_h)).unwrap();

    println!("Reconstructed A:\n{:?}", recon);

    let mut diff = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    diff.assign(&(&a - &recon)).unwrap();
    
    let norm = diff.norm();
    println!("Reconstruction Error Norm: {:e}", norm);
    
    if norm < 1e-12 {
        println!("LLT Verification: PASSED");
    } else {
        println!("LLT Verification: FAILED");
        std::process::exit(1);
    }
}

fn verify_ldlt() {
    println!("\n--- Verifying LDLT (L * D * L^H) ---");
    // Same matrix A
    let mut a = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    *a.get_mut(0, 0).unwrap() = Complex::new(4.0, 0.0);
    *a.get_mut(0, 1).unwrap() = Complex::new(0.0, 2.0);
    *a.get_mut(1, 0).unwrap() = Complex::new(0.0, -2.0);
    *a.get_mut(1, 1).unwrap() = Complex::new(5.0, 0.0);

    let ldlt = a.ldlt().unwrap();
    let l = ldlt.matrix_l();
    let d = ldlt.vector_d();
    println!("Matrix L:\n{:?}", l);
    println!("Vector D:\n{:?}", d);

    // Reconstruct A = P^T * L * D * L^H * P
    
    let mut d_mat = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    *d_mat.get_mut(0, 0).unwrap() = d.get(0).unwrap().clone();
    *d_mat.get_mut(1, 1).unwrap() = d.get(1).unwrap().clone();
    
    let l_h = adjoint(l);
    let mut temp = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    // temp = L * D_mat
    temp.assign(&(l * &d_mat)).unwrap();
    
    let mut recon = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    // recon = temp * L_h
    recon.assign(&(&temp * &l_h)).unwrap();
    
    // Pivoting: P * A * P^T = L * D * L^H  => A = P^T * L * D * L^H * P
    let p = ldlt.permutation();
    // For this simple matrix, likely no pivoting or trivial.
    // If p is [0, 1], then P is identity.
    println!("Permutation: {:?}", p);
    
    println!("Reconstructed (ignoring P):\n{:?}", recon);
    
    // Solve Ax=b check 
    let mut b = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    *b.get_mut(0, 0).unwrap() = Complex::new(2.0, 1.0);
    *b.get_mut(1, 0).unwrap() = Complex::new(1.0, 3.0);
    
    let x = ldlt.solve(&b).unwrap();
    
    // Check A * x = b
    let mut ax = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    ax.assign(&(&a * &x)).unwrap();
    
    let mut r = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    r.assign(&(&ax - &b)).unwrap();
    
    let norm = r.norm();
    println!("Residual Norm (Ax - b): {:e}", norm);
     if norm < 1e-12 {
        println!("LDLT Verification: PASSED");
    } else {
        println!("LDLT Verification: FAILED");
         std::process::exit(1);
    }
}

fn verify_qr() {
    println!("\n--- Verifying HouseholderQR (Q * R) ---");
    // A = [ 1+i  2 ]
    //     [ 3    4-i ]
    let mut a = Matrix::<Complex<f64>, _>::new_dynamic(2, 2).unwrap();
    *a.get_mut(0, 0).unwrap() = Complex::new(1.0, 1.0);
    *a.get_mut(0, 1).unwrap() = Complex::new(2.0, 0.0);
    *a.get_mut(1, 0).unwrap() = Complex::new(3.0, 0.0);
    *a.get_mut(1, 1).unwrap() = Complex::new(4.0, -1.0);

    let qr = a.householder_qr().unwrap();
    
    // Verify Ax = b solve.
    let mut b = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    *b.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
    *b.get_mut(1, 0).unwrap() = Complex::new(0.0, 1.0);
    
    let x = qr.solve(&b).unwrap();
    
    let mut ax = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    ax.assign(&(&a * &x)).unwrap();
    
    let mut r = Matrix::<Complex<f64>, _>::new_dynamic(2, 1).unwrap();
    r.assign(&(&ax - &b)).unwrap();
    
    let norm = r.norm();
    println!("Residual Norm (Ax - b): {:e}", norm);
    
    if norm < 1e-12 {
        println!("QR Verification: PASSED");
    } else {
         println!("QR Verification: FAILED");
         std::process::exit(1);
    }
}
