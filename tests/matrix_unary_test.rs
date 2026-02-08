
#[cfg(test)]
mod tests {
    use eigen_rs::core::matrix::{Matrix, DynamicStorage};
    
    use eigen_rs::core::xpr::MatrixXpr;

    #[test]
    fn test_unary_sin_cos() {
        let rows = 4;
        let cols = 4;
        let mut m = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(rows, cols).unwrap();
        
        // Fill with some angles
        for i in 0..rows {
            for j in 0..cols {
                *m.get_mut(i, j).unwrap() = (i + j) as f32 * 0.1;
            }
        }

        let s = m.sin();
        let c = m.cos();

        for i in 0..rows {
            for j in 0..cols {
                let val = *m.get(i, j).unwrap();
                let sin_val = s.eval(i, j);
                let cos_val = c.eval(i, j);

                assert!((sin_val - val.sin()).abs() < 1e-6, "Sin mismatch at {},{}", i, j);
                assert!((cos_val - val.cos()).abs() < 1e-6, "Cos mismatch at {},{}", i, j);
            }
        }
    }

    #[test]
    fn test_unary_exp_log() {
        let rows = 4;
        let cols = 4;
        let mut m = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(rows, cols).unwrap();
        
        // Fill with positive values
        for i in 0..rows {
            for j in 0..cols {
                *m.get_mut(i, j).unwrap() = (i + j) as f32 + 1.0;
            }
        }

        let e = m.exp();
        let l = m.ln();

        for i in 0..rows {
            for j in 0..cols {
                let val = *m.get(i, j).unwrap();
                let exp_val = e.eval(i, j);
                let ln_val = l.eval(i, j);

                assert!((exp_val - val.exp()).abs() < 1e-4, "Exp mismatch at {},{}", i, j); // Exp grows fast, loose tolerance
                assert!((ln_val - val.ln()).abs() < 1e-6, "Ln mismatch at {},{}", i, j);
            }
        }
    }
    
    #[test]
    fn test_unary_composition() {
        let mut m = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(2, 2).unwrap();
        *m.get_mut(0, 0).unwrap() = 1.0;
        
        // exp(ln(x)) = x
        let l = m.ln();
        
        // We can't chain directly yet because .ln() returns CwiseUnaryOp which is Xpr, 
        // but Matrix methods are implemented on Matrix<T, S>.
        // To chain, we need Xpr to impl the methods too.
        // For now, assign to temp.
        
        let mut temp = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(2, 2).unwrap();
        temp.assign(&l).unwrap();
        
        let e = temp.exp();
        
        let val = e.eval(0, 0);
        assert!((val - 1.0).abs() < 1e-6);
    }
}
