use eigen_rs::core::decompositions::bidiagonal::Bidiagonalization;
use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;

fn main() {
    let n = 32;
    let mut mat = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    for i in 0..n {
        for j in 0..n {
            *mat.get_mut(i, j).unwrap() = ((i + j) as f64).sin();
        }
    }

    let bidiag = Bidiagonalization::new(&mat).unwrap();
    let u = bidiag.matrix_u();
    let v = bidiag.matrix_v();

    // Check U^T U
    let ut = u.transpose();
    let mut prod_u = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    prod_u.assign(&(&ut * &u)).unwrap();

    let mut max_err_u = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = *prod_u.get(i, j).unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (val - expected).abs();
            if diff > max_err_u {
                max_err_u = diff;
            }
        }
    }
    println!("Bidiagonal U Orthogonality Error: {:.5e}", max_err_u);

    // Check V^T V
    let vt = v.transpose();
    let mut prod_v = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    prod_v.assign(&(&vt * &v)).unwrap();

    let mut max_err_v = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = *prod_v.get(i, j).unwrap();
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (val - expected).abs();
            if diff > max_err_v {
                max_err_v = diff;
            }
        }
    }
    println!("Bidiagonal V Orthogonality Error: {:.5e}", max_err_v);
}
