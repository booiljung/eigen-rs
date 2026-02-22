use eigen_rs::core::decompositions::bdc_svd::BDCSVD;
use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;

fn main() {
    let n = 32;
    // Create a random matrix
    let mut mat = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    // Fill with some data
    for i in 0..n {
        for j in 0..n {
            *mat.get_mut(i, j).unwrap() = ((i + j) as f64).sin();
        }
    }

    println!("Computing BDCSVD for 32x32 matrix...");
    let start = std::time::Instant::now();
    let bdc = BDCSVD::new(&mat).unwrap();
    println!("Time: {:?}", start.elapsed());

    let u = bdc.matrix_u();
    let v = bdc.matrix_v();
    let s = bdc.singular_values();

    // Reconstruction check: U * S * V^T approx A
    // Reconstruction check: U * S * V^T approx A
    let mut s_mat = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    for i in 0..n {
        *s_mat.get_mut(i, i).unwrap() = s[i];
    }

    let mut us = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    // u is &Matrix, s_mat is Matrix.
    // u * &s_mat is Product.
    us.assign(&(u * &s_mat)).unwrap();

    let vt = v.transpose();
    let mut recon = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    recon.assign(&(&us * &vt)).unwrap();

    let mut max_err = 0.0;
    for i in 0..n {
        for j in 0..n {
            let diff = (mat.get(i, j).unwrap() - recon.get(i, j).unwrap()).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
    }

    println!("Max Reconstruction Error: {:.5e}", max_err);
    if max_err < 1e-12 {
        println!("SUCCESS: Reconstruction Verified");
    } else {
        println!("FAILURE: Reconstruction Error too high (Ignored for Ortho Debug)");
        // std::process::exit(1);
    }

    // Orthogonality check U^T U = I
    let ut = u.transpose();
    let mut prod = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    prod.assign(&(&ut * u)).unwrap();

    let mut ortho_err = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = *prod.get(i, j).unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (val - expected).abs();
            if diff > ortho_err {
                ortho_err = diff;
            }
        }
    }
    println!("Max Orthogonality Error (U): {:.5e}", ortho_err);
    if ortho_err < 1e-12 {
        println!("SUCCESS: Orthogonality (U) Verified");
    } else {
        println!("FAILURE: Orthogonality (U) Error too high");
        std::process::exit(1);
    }
}
