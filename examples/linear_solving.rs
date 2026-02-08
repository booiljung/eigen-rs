
use eigen_rs::core::matrix::{Matrix, DynamicStorage};
use eigen_rs::core::decompositions::lu::PartialPivLU;
use eigen_rs::core::decompositions::qr::HouseholderQR;
use eigen_rs::core::decompositions::ldlt::LDLT;

fn main() -> Result<(), String> {
    println!("=== Linear Solving with eigen-rs ===");

    // 1. LU Decomposition (Ax = b)
    println!("\n--- LU Decomposition ---");
    let size = 4;
    let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, size).unwrap();
    // Fill with some data
    for i in 0..size {
        for j in 0..size {
            *a.get_mut(i, j).unwrap() = (i as f64 + j as f64).sin();
        }
        *a.get_mut(i, i).unwrap() += 2.0; // Make diagonally dominant-ish
    }
    
    let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    for i in 0..size {
        *b.get_mut(i, 0).unwrap() = i as f64;
    }

    let lu = PartialPivLU::new(&a)?;
    let x_lu = lu.solve(&b)?;
    
    // Verify A * x = b
    let mut ax = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    ax.assign(&(&a * &x_lu)).unwrap();
    let mut diff = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    diff.assign(&(&ax - &b)).unwrap();
    println!("Norm of (Ax - b) using LU: {:.6e}", diff.norm());

    // 2. QR Decomposition (Least Squares)
    println!("\n--- QR Decomposition ---");
    let qr = HouseholderQR::new(&a)?;
    let x_qr = qr.solve(&b)?; // For square A, equivalent to exact solution

    let mut ax_qr = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    ax_qr.assign(&(&a * &x_qr)).unwrap();
    diff.assign(&(&ax_qr - &b)).unwrap();
    println!("Norm of (Ax - b) using QR: {:.6e}", diff.norm());

    // 3. Cholesky Decomposition (Ax = b, A is SPD)
    println!("\n--- Cholesky Decomposition (LDLT) ---");
    // Generate SPD matrix: M = A^T * A
    let at = a.transpose();
    
    // Need explicit intermediate var or correct chaining
    let mut spd_mat = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, size).unwrap();
    spd_mat.assign(&(&at * &a)).unwrap(); 

    let ldlt = LDLT::new(&spd_mat)?;
    let x_ldlt = ldlt.solve(&b)?;

    let mut mx_ldlt = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, 1).unwrap();
    mx_ldlt.assign(&(&spd_mat * &x_ldlt)).unwrap();
    diff.assign(&(&mx_ldlt - &b)).unwrap();
    println!("Norm of (Mx - b) using LDLT: {:.6e}", diff.norm());

    // 4. Eigensolver
    println!("\n--- Self-Adjoint Eigensolver ---");
    // Use the SPD matrix from above
    let solver = eigen_rs::core::decompositions::SelfAdjointEigenSolver::new(&spd_mat, true)?;
    println!("Eigenvalues:\n{:?}", solver.eigenvalues());
    
    Ok(())
}
