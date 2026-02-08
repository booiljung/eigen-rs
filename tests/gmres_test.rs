
#[cfg(test)]
mod tests {
    use eigen_rs::core::sparse::solvers::GMRES;
    use eigen_rs::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::matrix::{Matrix, DynamicStorage};
    use eigen_rs::core::sparse::solvers::Preconditioner;
    use eigen_rs::core::sparse::solvers::iterative_solver_base::IdentityPreconditioner;
    use eigen_rs::core::sparse::solvers::iterative_solver_base::DiagonalPreconditioner;

    #[test]
    fn test_gmres_simple() {
        // Solve A x = b
        // A = [4 -1; -1 4] (Symmetric Positive Definite, but GMRES handles general)
        // b = [3; 3]
        // x = [1; 1]
        
        let mut a = SparseMatrix::<f64>::new(2, 2, StorageOrder::RowMajor);
        a.set_from_triplets(vec![
            Triplet::new(0, 0, 4.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, 4.0),
        ]);
        
        let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(2, 1).unwrap();
        *b.get_mut(0, 0).unwrap() = 3.0;
        *b.get_mut(1, 0).unwrap() = 3.0;
        
        let mut solver = GMRES::<f64, IdentityPreconditioner>::new();
        let x = solver.solve(&a, &b).unwrap();
        
        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-4);
        assert!((x.get(1, 0).unwrap() - 1.0).abs() < 1e-4);
        println!("Iterations: {}", solver.iterations());
    }
    
    #[test]
    fn test_gmres_non_symmetric() {
        // A = [1 2; 0 3]
        // b = [5; 6]
        // x = [1; 2]  (1*1 + 2*2 = 5; 3*2 = 6)
        
        let mut a = SparseMatrix::<f64>::new(2, 2, StorageOrder::RowMajor);
        a.set_from_triplets(vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 1, 3.0),
        ]);
        
        let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(2, 1).unwrap();
        *b.get_mut(0, 0).unwrap() = 5.0;
        *b.get_mut(1, 0).unwrap() = 6.0;
        
        let mut solver = GMRES::<f64, IdentityPreconditioner>::new();
        // Since it's only 2x2, restart won't matter much.
        let x = solver.solve(&a, &b).unwrap();
        
        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-4);
        assert!((x.get(1, 0).unwrap() - 2.0).abs() < 1e-4);
    }
    
    #[test]
    fn test_gmres_preconditioned() {
         // A = [10 1; 1 10]
         // Diagonal Preconditioner M = [1/10 0; 0 1/10]
         // b = [11; 11] -> x = [1; 1]
         
        let mut a = SparseMatrix::<f64>::new(2, 2, StorageOrder::RowMajor);
        a.set_from_triplets(vec![
            Triplet::new(0, 0, 10.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 10.0),
        ]);
        
        let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(2, 1).unwrap();
        *b.get_mut(0, 0).unwrap() = 11.0;
        *b.get_mut(1, 0).unwrap() = 11.0;
        
        let mut solver = GMRES::<f64, DiagonalPreconditioner<f64>>::new();
        let x = solver.solve(&a, &b).unwrap();
        
        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-4);
        assert!((x.get(1, 0).unwrap() - 1.0).abs() < 1e-4);
    }
}
