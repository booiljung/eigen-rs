
use eigen_rs::core::sparse::sparse_matrix::{SparseMatrix, Triplet, StorageOrder};
use eigen_rs::core::sparse::solvers::SparseLU;
use eigen_rs::core::sparse::solvers::bicgstab::BiCGSTAB;
use eigen_rs::core::matrix::{Matrix, DynamicStorage};

fn main() -> Result<(), String> {
    println!("=== Sparse Solving with eigen-rs ===");

    // 1. Create Sparse Matrix (COO -> CSR)
    let size = 100;
    println!("Creating {}x{} sparse matrix...", size, size);
    
    // Tridiagonal matrix (-1, 2, -1)
    let nnz = 3 * size - 2;
    let mut triplets = Vec::with_capacity(nnz);
    for i in 0..size {
        triplets.push(Triplet::new(i, i, 2.0));
        if i > 0 { triplets.push(Triplet::new(i, i-1, -1.0)); }
        if i < size - 1 { triplets.push(Triplet::new(i, i+1, -1.0)); }
    }
    
    let mut a = SparseMatrix::<f64>::new(size, size, StorageOrder::ColMajor);
    a.set_from_triplets(triplets);
    
    // Create RHS
    let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    // b = [1, 1, ..., 1]^T
    for i in 0..size { *b.get_mut(i, 0).unwrap() = 1.0; }

    // 2. Direct Solver: SparseLU
    println!("\n--- SparseLU Direct Solver ---");
    let mut solver_lu = SparseLU::new();
    solver_lu.compute(&a)?;
    let x_lu = solver_lu.solve(&b)?;

    // Verify: A * x
    let ax_lu = a.mul_dense(&x_lu)?;
    let mut diff = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    diff.assign(&(&ax_lu - &b)).unwrap();
    println!("Norm of (Ax - b) using SparseLU: {:.6e}", diff.norm());

    // 3. Iterative Solver: BiCGSTAB
    println!("\n--- BiCGSTAB Iterative Solver ---");
    let mut solver_bicg = BiCGSTAB::new();
    let x_bicg = solver_bicg.solve(&a, &b)?;
    let ax_bicg = a.mul_dense(&x_bicg)?;
    diff.assign(&(&ax_bicg - &b)).unwrap();
    println!("Norm of (Ax - b) using BiCGSTAB: {:.6e}", diff.norm());

    // 4. Iterative Solver: Conjugate Gradient (CG) - For SPD matrices
    // The tridiagonal matrix (-1, 2, -1) IS Symmetric Positive Definite!
    println!("\n--- Conjugate Gradient Iterative Solver ---");
    let mut solver_cg = eigen_rs::core::sparse::solvers::conjugate_gradient::ConjugateGradient::new();
    let x_cg = solver_cg.solve(&a, &b)?; // Note: solve takes &a, &b directly
    let ax_cg = a.mul_dense(&x_cg)?;
    diff.assign(&(&ax_cg - &b)).unwrap();
    println!("Norm of (Ax - b) using CG: {:.6e}", diff.norm());
    
    Ok(())
}
