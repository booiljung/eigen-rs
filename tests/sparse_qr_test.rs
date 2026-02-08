#[cfg(test)]
mod tests {
    use eigen_rs::core::sparse::sparse_matrix::SparseMatrix;
    use eigen_rs::core::sparse::solvers::sparse_qr::SparseQR;
    use eigen_rs::core::matrix::Matrix;
    use eigen_rs::core::storage::DynamicStorage;
    use eigen_rs::core::sparse::Triplet;
    use eigen_rs::core::sparse::iterators::InnerIterator;

    #[test]
    fn test_sparse_qr_identity() {
        let mut triplets = Vec::new();
        triplets.push(Triplet::new(0, 0, 1.0));
        triplets.push(Triplet::new(1, 1, 1.0));
        triplets.push(Triplet::new(2, 2, 1.0));
        
        let mut mat = SparseMatrix::<f64>::new(3, 3, eigen_rs::core::sparse::StorageOrder::ColMajor);
        mat.set_from_triplets(triplets);

        let mut qr = SparseQR::new();
        qr.compute(&mat).unwrap();

        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let x = qr.solve(&b).unwrap();

        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(1, 0).unwrap() - 2.0).abs() < 1e-10);
        assert!((x.get(2, 0).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_qr_diagonal() {
        let mut triplets = Vec::new();
        triplets.push(Triplet::new(0, 0, 2.0));
        triplets.push(Triplet::new(1, 1, 4.0));
        triplets.push(Triplet::new(2, 2, 8.0));
        
        let mut mat = SparseMatrix::<f64>::new(3, 3, eigen_rs::core::sparse::StorageOrder::ColMajor);
        mat.set_from_triplets(triplets);

        let mut qr = SparseQR::new();
        qr.compute(&mat).unwrap();

        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![2.0, 4.0, 8.0]).unwrap();
        let x = qr.solve(&b).unwrap();

        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(2, 0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_qr_rectangular_least_squares() {
        // A = [1 0]
        //     [0 1]
        //     [0 0]
        // b = [1, 2, 3]
        // x should be [1, 2]
        
        let mut triplets = Vec::new();
        triplets.push(Triplet::new(0, 0, 1.0));
        triplets.push(Triplet::new(1, 1, 1.0));
        
        let mut mat = SparseMatrix::<f64>::new(3, 2, eigen_rs::core::sparse::StorageOrder::ColMajor);
        mat.set_from_triplets(triplets);
        
        let mut qr = SparseQR::new();
        qr.compute(&mat).unwrap();
        
         let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
         let x = qr.solve(&b).unwrap();
         
         assert_eq!(x.rows(), 2);
         assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
         assert!((x.get(1, 0).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_qr_tridiagonal() {
        // 4x4 Tridiagonal
        // [ 2 -1  0  0 ]
        // [-1  2 -1  0 ]
        // [ 0 -1  2 -1 ]
        // [ 0  0 -1  2 ]
        // Solution to Ax = [1,0,0,1] is symmetric.
        
        let n = 4;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 2.0));
            if i > 0 { triplets.push(Triplet::new(i, i-1, -1.0)); }
            if i < n-1 { triplets.push(Triplet::new(i, i+1, -1.0)); }
        }
        
        let mut mat = SparseMatrix::<f64>::new(n, n, eigen_rs::core::sparse::StorageOrder::ColMajor);
        mat.set_from_triplets(triplets);
        
        let mut qr = SparseQR::new();
        qr.compute(&mat).expect("QR compute failed");
        
        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(n, 1, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let x = qr.solve(&b).expect("QR solve failed");
        
        // Verify Ax = b
        // Verify Ax = b
        // Manual multiplication mat * x
        let mut result = vec![0.0; n];
        for k in 0..mat.outer_size() {
            let mut it = InnerIterator::new(&mat, k);
            while it.is_valid() {
                 let r = it.row();
                 let c = it.col();
                 let val = it.value();
                 result[r] += val * x.get(c, 0).unwrap();
                 it.next();
            }
        }
        
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }
}
