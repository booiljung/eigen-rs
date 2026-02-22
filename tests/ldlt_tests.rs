#[cfg(test)]
mod tests {
    use eigen_rs::core::decompositions::LDLT;
    use eigen_rs::core::matrix::Matrix;
    use eigen_rs::core::storage::DynamicStorage;

    #[test]
    fn test_ldlt_decomposition_dynamic() {
        let n = 4;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        // Construct a symmetric positive definite matrix
        // A = L D L^T
        // L = [1 0 0]
        //     [2 1 0]
        //     [3 4 1]
        // D = [1 2 3]
        // L D L^T ... let's just make a simple SPD matrix
        // [ 4  1 -2  2]
        // [ 1  2  0  1]
        // [-2  0  3 -2]
        // [ 2  1 -2 -1] is indifferent/indefinite maybe?
        // Let's use the one from Tridiagonal test or generate essentially
        let vals = vec![
            4.0, 1.0, -2.0, 2.0, 1.0, 5.0, 1.0, 0.0, -2.0, 1.0, 10.0, -1.0, 2.0, 0.0, -1.0, 5.0,
        ];
        // Ensure symmetry
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = vals[i * n + j];
            }
        }

        let ldlt = LDLT::new(&a).unwrap();

        let l = ldlt.matrix_l();
        let d = ldlt.vector_d();
        let p = ldlt.permutation();

        // Verify Reconstruction: P^T L D L^T P = A
        // Or P A P^T = L D L^T

        // Let's compute P A P^T
        let mut p_a_pt = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        // Apply permutation to A
        let mut pa = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                *pa.get_mut(i, j).unwrap() = *a.get(p[i], j).unwrap();
            }
        }
        for i in 0..n {
            for j in 0..n {
                *p_a_pt.get_mut(i, j).unwrap() = *pa.get(i, p[j]).unwrap();
            }
        }

        // Compute L * D * L^T
        let mut ldlt_res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        let mut l_d = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                *l_d.get_mut(i, j).unwrap() = *l.get(i, j).unwrap() * d[j];
            }
        }

        ldlt_res.assign_product(&(&l_d * &l.transpose())).unwrap();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ldlt_res.get(i, j).unwrap() - p_a_pt.get(i, j).unwrap()).abs() < 1e-9,
                    "Mismatch at ({},{}): expected {}, got {}",
                    i,
                    j,
                    p_a_pt.get(i, j).unwrap(),
                    ldlt_res.get(i, j).unwrap()
                );
            }
        }
    }
}
