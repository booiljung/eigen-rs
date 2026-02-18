extern crate eigen_rs;
use eigen_rs::core::iterative_solvers::ConjugateGradient;
use eigen_rs::core::iterative_solvers::traits::IterativeSolver;
use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::storage::DynamicStorage;

fn main() {
    let n = 50;
    
    // 1. Generate a random matrix A
    let mut a_rand = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    for i in 0..n {
        for j in 0..n {
            *a_rand.get_mut(i, j).unwrap() = (i + j) as f64 * 0.1;
        }
    }
    
    // 2. Make it Symmetric Positive Definite (A = B * B^T + Identity)
    let a_t = a_rand.transpose();
    // Evaluate A = A^T * A
    let mut a = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    a.assign(&(&a_t * &a_rand)).unwrap();
    
    // Add Identity to ensure positive definiteness margin
    for i in 0..n {
        *a.get_mut(i, i).unwrap() += 1.0;
    }

    // 3. Generate a known solution x_ref
    let mut x_ref = MatrixX::<f64>::new_dynamic(n, 1).unwrap();
    for i in 0..n {
        *x_ref.get_mut(i, 0).unwrap() = i as f64;
    }

    // 4. Compute b = A * x_ref
    let mut b = MatrixX::<f64>::new_dynamic(n, 1).unwrap();
    b.assign(&(&a * &x_ref)).unwrap();

    // 5. Solve using Conjugate Gradient
    println!("Initializing CG Solver...");
    let mut cg = ConjugateGradient::<f64, MatrixX<f64>>::new();
    
    cg.compute(&a);
    
    println!("Solving...");
    let x = cg.solve(&b);

    println!("Iterations: {}", cg.iterations());
    println!("Estimated Error: {:.5e}", cg.error());

    // 6. Verify Result
    let mut diff = MatrixX::<f64>::new_dynamic(n, 1).unwrap();
    diff.assign(&(&x - &x_ref)).unwrap();
    
    let error = diff.norm();
    println!("Actual Error ||x - x_ref||: {:.5e}", error);

    if error < 1e-5 {
        println!("Dense Test PASSED!");
    } else {
        println!("Dense Test FAILED!");
        std::process::exit(1);
    }
    
    // ==========================================
    // Sparse Matrix Test
    // ==========================================
    println!("\n=== Sparse Matrix Test ===");
    use eigen_rs::core::sparse::{SparseMatrix, Triplet, StorageOrder};
    
    // Reuse the same A but convert to Sparse
    let mut a_sparse = SparseMatrix::<f64>::new(n, n, StorageOrder::RowMajor);
    let mut triplets = Vec::new();

    // Fill sparse matrix from dense A
    // Note: This is efficient enough for small N=50
    for i in 0..n {
        for j in 0..n {
            let val = *a.get(i, j).unwrap();
            if val.abs() > 1e-10 {
                triplets.push(Triplet::new(i, j, val));
            }
        }
    }
    a_sparse.set_from_triplets(triplets);
    
    println!("Initializing CG Solver (Sparse)...");
    let mut cg_sparse = ConjugateGradient::<f64, SparseMatrix<f64>>::new();
    cg_sparse.compute(&a_sparse);
    
    println!("Solving Sparse System...");
    let x_sparse = cg_sparse.solve(&b); // b is still dense vector
    
    println!("Iterations: {}", cg_sparse.iterations());
    println!("Estimated Error: {:.5e}", cg_sparse.error());
    
    let mut diff_sparse = MatrixX::<f64>::new_dynamic(n, 1).unwrap();
    diff_sparse.assign(&(&x_sparse - &x_ref)).unwrap();
    let error_sparse = diff_sparse.norm();
    println!("Sparse Actual Error: {:.5e}", error_sparse);
    
    if error_sparse < 1e-5 {
        println!("Sparse Test PASSED!");
    } else {
        println!("Sparse Test FAILED!");
        std::process::exit(1);
    }
    
    use eigen_rs::core::iterative_solvers::BiCGSTAB;

    // ==========================================
    // BiCGSTAB Test (Sparse, Non-Symmetric potentially)
    // ==========================================
    println!("\n=== BiCGSTAB Test ===");
    // Using same A (SPD) for simplicity, but BiCGSTAB works for general matrices
    
    println!("Initializing BiCGSTAB Solver...");
    let mut bicg = BiCGSTAB::<f64, SparseMatrix<f64>>::new();
    bicg.compute(&a_sparse);
    
    println!("Solving...");
    let x_bicg = bicg.solve(&b);
    
    println!("Iterations: {}", bicg.iterations());
    println!("Estimated Error: {:.5e}", bicg.error());
    
    let mut diff_bicg = MatrixX::<f64>::new_dynamic(n, 1).unwrap();
    diff_bicg.assign(&(&x_bicg - &x_ref)).unwrap();
    let error_bicg = diff_bicg.norm();
    println!("BiCGSTAB Error: {:.5e}", error_bicg);
    
    if error_bicg < 1e-5 {
        println!("BiCGSTAB Test PASSED!");
    } else {
        println!("BiCGSTAB Test FAILED!");
        std::process::exit(1);
    }
}
