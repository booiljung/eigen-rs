use eigen_rs::core::decompositions::Tridiagonalization;
use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;

fn main() {
    let n: usize = 4;
    let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    // Symmetric matrix
    let val = vec![
        4.0, 1.0, -2.0, 2.0,
        1.0, 2.0, 0.0, 1.0,
        -2.0, 0.0, 3.0, -2.0,
        2.0, 1.0, -2.0, -1.0
    ];
    
    for i in 0..n {
        for j in 0..n {
            *a.get_mut(i, j).unwrap() = val[i * n + j];
        }
    }

    println!("Original A:");
    print_mat(&a);

    let tridiag = Tridiagonalization::new(&a).unwrap();
    let t = tridiag.matrix_t();
    let q = tridiag.matrix_q();

    println!("T:");
    print_mat(&t);
    println!("Q:");
    print_mat(&q);

    let qt = q.transpose();
    
    let mut q_qt_res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    q_qt_res.assign_product(&(&q * &qt)).unwrap();
    println!("Q * Q^T (Identity?):");
    print_mat(&q_qt_res);

    let mut t_qt_res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    t_qt_res.assign_product(&(&t * &qt)).unwrap();
    println!("T * Q^T:");
    print_mat(&t_qt_res);
    
    let mut recon = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    recon.assign_product(&(&q * &t_qt_res)).unwrap();

    println!("Recontructed A (Q * T * Q^T):");
    print_mat(&recon);

    // GEMM Debug
    println!("--- GEMM Debug ---");
    let ident = Matrix::<f64, DynamicStorage<f64>>::identity(n, n);
    let mut debug_res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
    debug_res.assign_product(&(&q * &ident)).unwrap();
    println!("Q * I:");
    print_mat(&debug_res);
    
    // Check if Q * I == Q
    for i in 0..n {
        for j in 0..n {
            if (debug_res.get(i, j).unwrap() - q.get(i, j).unwrap()).abs() > 1e-10 {
                println!("GEMM Fail: (Q*I)({}, {}) = {}, Expected {}", i, j, debug_res.get(i, j).unwrap(), q.get(i, j).unwrap());
            }
        }
    }
    println!("--- End GEMM Debug ---");

    for i in 0..n {
        for j in 0..n {
            let diff = (recon.get(i, j).unwrap() - a.get(i, j).unwrap()).abs();
            if diff > 1e-10 {
                println!("Mismatch at ({}, {}): A={}, Recon={}, Diff={}", i, j, a.get(i, j).unwrap(), recon.get(i, j).unwrap(), diff);
            }
        }
    }
}

fn print_mat(m: &Matrix<f64, DynamicStorage<f64>>) {
    let rows = m.rows();
    let cols = m.cols();
    for i in 0..rows {
        print!("[");
        for j in 0..cols {
            print!("{:8.4} ", m.get(i, j).unwrap());
        }
        println!("]");
    }
}
