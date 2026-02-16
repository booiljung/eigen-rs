#[cfg(test)]
mod tests {
    use eigen_rs::core::matrix::Matrix;
    use eigen_rs::core::storage::DynamicStorage;
    use eigen_rs::core::decompositions::HessenbergDecomposition;

    #[test]
    fn test_hessenberg_decomposition_perf_correctness() {
        let n = 28; // Small but non-trivial
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        // Deterministic 'random'
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = (i as f64 * 1.3 + j as f64 * 0.7).cos();
            }
        }

        let decomp = HessenbergDecomposition::new(&a).unwrap();
        
        // A = Q H Q^T -> Q^T A Q = H
        let h = decomp.matrix_h();
        let q = decomp.matrix_q();

        // 1. Check Q orthogonality
        let qt = q.transpose();
        let mut qtq = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        qtq.assign_product(&(&qt * &q)).unwrap();

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq.get(i, j).unwrap() - expected).abs() < 1e-9, "Q Orthogonality failed at ({},{})", i, j);
            }
        }
        
        // 2. Check Reconstruction: Q H Q^T = A
        let mut h_qt = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        h_qt.assign_product(&(&h * &qt)).unwrap();
        let mut recon = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        recon.assign_product(&(&q * &h_qt)).unwrap();

        for i in 0..n {
            for j in 0..n {
                let diff = (recon.get(i, j).unwrap() - a.get(i, j).unwrap()).abs();
                assert!(diff < 1e-9, "Reconstruction failed at ({},{}) expected {}, got {}", i, j, a.get(i, j).unwrap(), recon.get(i, j).unwrap());
            }
        }

        // 3. Check H is upper Hessenberg
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    assert!(h.get(i, j).unwrap().abs() < 1e-9, "H not Hessenberg at ({},{})", i, j);
                }
            }
        }
    }
}
