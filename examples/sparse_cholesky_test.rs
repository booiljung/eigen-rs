use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::sparse::solvers::{SimplicialLDLT, SimplicialLLT};
use eigen_rs::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder, Triplet};

fn main() {
    println!("=== Sparse Cholesky Verification ===");

    // 1. Define a simple SPD matrix (Poisson 1D / Tridiagonal)
    // [ 2 -1  0 ]
    // [-1  2 -1 ]
    // [ 0 -1  2 ]
    let size = 10;
    let mut triplets = Vec::new();
    for i in 0..size {
        triplets.push(Triplet::new(i, i, 2.0));
        if i > 0 {
            triplets.push(Triplet::new(i, i - 1, -1.0));
            triplets.push(Triplet::new(i - 1, i, -1.0));
        }
    }

    let mut a = SparseMatrix::<f64>::new(size, size, StorageOrder::ColMajor);
    a.set_from_triplets(triplets);

    // 2. Define RHS b = A * x_ref
    // Let x_ref = [1, 1, ..., 1]
    let mut x_ref = MatrixX::<f64>::new_dynamic(size, 1).unwrap();
    for i in 0..size {
        *x_ref.get_mut(i, 0).unwrap() = 1.0;
    }

    // b = A * x_ref
    let b = (&a * &x_ref).unwrap();

    // 3. Test SimplicialLLT
    println!("\nTesting SimplicialLLT...");
    let mut llt = SimplicialLLT::new();
    if let Err(e) = llt.compute(&a) {
        println!("LLT Compute Failed: {}", e);
        std::process::exit(1);
    }

    let x_llt = llt.solve(&b).unwrap();
    let mut diff_llt_mat = MatrixX::<f64>::new_dynamic(size, 1).unwrap();
    diff_llt_mat.assign(&(&x_llt - &x_ref)).unwrap();
    let diff_llt = diff_llt_mat.norm();
    println!("LLT Error ||x - x_ref||: {:e}", diff_llt);

    if diff_llt < 1e-10 {
        println!("SimplicialLLT Test PASSED!");
    } else {
        println!("SimplicialLLT Test FAILED!");
        std::process::exit(1);
    }

    // 4. Test SimplicialLDLT
    println!("\nTesting SimplicialLDLT...");
    let mut ldlt = SimplicialLDLT::new();
    if let Err(e) = ldlt.compute(&a) {
        println!("LDLT Compute Failed: {}", e);
        std::process::exit(1);
    }

    let x_ldlt = ldlt.solve(&b).unwrap();
    let mut diff_ldlt_mat = MatrixX::<f64>::new_dynamic(size, 1).unwrap();
    diff_ldlt_mat.assign(&(&x_ldlt - &x_ref)).unwrap();
    let diff_ldlt = diff_ldlt_mat.norm();
    println!("LDLT Error ||x - x_ref||: {:e}", diff_ldlt);

    if diff_ldlt < 1e-10 {
        println!("SimplicialLDLT Test PASSED!");
    } else {
        println!("SimplicialLDLT Test FAILED!");
        std::process::exit(1);
    }
}
