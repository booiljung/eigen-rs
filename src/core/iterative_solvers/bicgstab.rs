use crate::core::iterative_solvers::traits::IterativeSolver;
use crate::core::matrix::{Matrix, MatrixX};
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};
use crate::core::sparse::SparseMatrix;
use std::marker::PhantomData;

/// BiConjugate Gradient Stabilized solver for non-self-adjoint problems.
pub struct BiCGSTAB<T: Scalar, M> {
    max_iterations: usize,
    tolerance: T,
    iterations: std::cell::RefCell<usize>,
    error: std::cell::RefCell<T>,
    matrix: Option<M>, // Store the matrix A
    _phantom: PhantomData<T>,
}

impl<T: Scalar, M: Clone> BiCGSTAB<T, M> {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: T::from_f64(1e-12),
            iterations: std::cell::RefCell::new(0),
            error: std::cell::RefCell::new(T::zero()),
            matrix: None,
            _phantom: PhantomData,
        }
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }
}

// Implementation for Dense Matrix
impl<T: Scalar<Real = T> + PartialOrd + 'static, S: Storage<T> + 'static> IterativeSolver<T, DynamicStorage<T>> 
    for BiCGSTAB<T, Matrix<T, S>> 
    where S: Clone 
{
    type MatrixType = Matrix<T, S>;

    fn compute(&mut self, matrix: &Self::MatrixType) {
        self.matrix = Some(matrix.clone());
    }

    fn solve<S2>(&self, b: &Matrix<T, S2>) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
    {
        let rows = b.rows();
        let cols = b.cols();
        let mut x0 = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        x0.set_zero();
        self.solve_with_guess(b, &x0)
    }

    fn solve_with_guess<S2, S3>(
        &self,
        b: &Matrix<T, S2>,
        x0: &Matrix<T, S3>,
    ) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
        S3: Storage<T>,
    {
        let a = self.matrix.as_ref().expect("Matrix not initialized. Call compute() first.");
        let rows = b.rows();
        let cols = b.cols();
        
        // x = x0
        let mut x = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        x.assign(x0).unwrap();

        // r = b - A * x
        let mut ax = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        ax.assign(&(a * &x)).unwrap();
        
        let mut r = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        r.assign(&(b - &ax)).unwrap();
        
        // r_hat = r
        let mut r_hat = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        r_hat.assign(&r).unwrap();
        
        // p = r
        let mut p = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        p.assign(&r).unwrap();
        
        // v = 0
        let mut v = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        v.set_zero();
        
        // s = 0 (placeholder)
        let mut s = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        
        // t = 0 (placeholder)
        let mut t = MatrixX::<T>::new_dynamic(rows, cols).unwrap();

        let mut rho_old = T::from_f64(1.0);
        let mut alpha = T::from_f64(1.0);
        let mut omega = T::from_f64(1.0);

        let tol_sq = self.tolerance * self.tolerance;
        let mut error_sq = r.squared_norm();
        
        *self.iterations.borrow_mut() = 0;
        *self.error.borrow_mut() = error_sq.sqrt();

        if error_sq < tol_sq {
            return x;
        }

        for i in 0..self.max_iterations {
            *self.iterations.borrow_mut() = i + 1;
            
            let rho = r_hat.dot(&r);
            if rho.abs() < T::from_f64(1e-30) {
                 break; 
            }
            
            let beta = (rho / rho_old) * (alpha / omega);
            
            // p = r + beta * (p - omega * v)
            let size = x.size();
            {
                let p_data = p.storage_mut().data_mut();
                let r_data = r.storage().data();
                let v_data = v.storage().data();
                for k in 0..size {
                    p_data[k] = r_data[k] + beta * (p_data[k] - omega * v_data[k]);
                }
            }
            
            // v = A * p
            v.assign(&(a * &p)).unwrap();
            
            let r_hat_v = r_hat.dot(&v);
             if r_hat_v.abs() < T::from_f64(1e-30) {
                 break; 
            }
            
            alpha = rho / r_hat_v;
            
            // s = r - alpha * v
            {
                let s_data = s.storage_mut().data_mut();
                let r_data = r.storage().data();
                let v_data = v.storage().data();
                for k in 0..size {
                    s_data[k] = r_data[k] - alpha * v_data[k];
                }
            }
            
            let s_norm_sq = s.squared_norm();
            if s_norm_sq < tol_sq {
                // x = x + alpha * p
                let x_data = x.storage_mut().data_mut();
                let p_data = p.storage().data();
                for k in 0..size {
                    x_data[k] += alpha * p_data[k];
                }
                *self.error.borrow_mut() = s_norm_sq.sqrt();
                break;
            }
            
            // t = A * s
            t.assign(&(a * &s)).unwrap();
            
            // omega = (t . s) / (t . t)
            let t_t = t.squared_norm();
            if t_t.abs() < T::from_f64(1e-30) {
                 break; 
            }
            omega = t.dot(&s) / t_t;
            
            // x = x + alpha * p + omega * s
            // r = s - omega * t
            {
                let x_data = x.storage_mut().data_mut();
                let p_data = p.storage().data();
                let s_data = s.storage().data();
                
                let r_data = r.storage_mut().data_mut();
                let t_data = t.storage().data();
                
                for k in 0..size {
                    x_data[k] += alpha * p_data[k] + omega * s_data[k];
                    r_data[k] = s_data[k] - omega * t_data[k];
                }
            }
            
            error_sq = r.squared_norm();
            *self.error.borrow_mut() = error_sq.sqrt();
            
            if error_sq < tol_sq {
                break;
            }
            
            if omega.abs() < T::from_f64(1e-30) {
                 break; 
            }
            
            rho_old = rho;
        }
        
        x
    }
    
    fn iterations(&self) -> usize {
        *self.iterations.borrow()
    }
    
    fn error(&self) -> T {
        *self.error.borrow()
    }
}

// Implementation for Sparse Matrix
impl<T: Scalar<Real = T> + PartialOrd + 'static> IterativeSolver<T, DynamicStorage<T>> 
    for BiCGSTAB<T, SparseMatrix<T>> 
{
    type MatrixType = SparseMatrix<T>;

    fn compute(&mut self, matrix: &Self::MatrixType) {
        self.matrix = Some(matrix.clone());
    }

    fn solve<S2>(&self, b: &Matrix<T, S2>) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
    {
        let rows = b.rows();
        let cols = b.cols();
        let mut x0 = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        x0.set_zero();
        self.solve_with_guess(b, &x0)
    }

    fn solve_with_guess<S2, S3>(
        &self,
        b: &Matrix<T, S2>,
        x0: &Matrix<T, S3>,
    ) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
        S3: Storage<T>,
    {
        let a = self.matrix.as_ref().expect("Matrix not initialized. Call compute() first.");
        let rows = b.rows();
        let cols = b.cols();
        
        // x = x0
        let mut x = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        x.assign(x0).unwrap();

        // r = b - A * x
        let ax = a.mul_dense(&x).expect("Sparse-Dense multiplication failed");
        
        let mut r = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        r.assign(&(b - &ax)).unwrap();
        
        // r_hat = r
        let mut r_hat = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        r_hat.assign(&r).unwrap();
        
        // p = r
        let mut p = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        p.assign(&r).unwrap();
        
        // v = 0
        let mut v = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        v.set_zero();
        
        // s = 0 (placeholder)
        let mut s = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        
        // t = 0 (placeholder)
        let mut t = MatrixX::<T>::new_dynamic(rows, cols).unwrap();

        let mut rho_old = T::from_f64(1.0);
        let mut alpha = T::from_f64(1.0);
        let mut omega = T::from_f64(1.0);

        let tol_sq = self.tolerance * self.tolerance;
        let mut error_sq = r.squared_norm();
        
        *self.iterations.borrow_mut() = 0;
        *self.error.borrow_mut() = error_sq.sqrt();

        if error_sq < tol_sq {
            return x;
        }

        for i in 0..self.max_iterations {
            *self.iterations.borrow_mut() = i + 1;
            
            let rho = r_hat.dot(&r);
            if rho.abs() < T::from_f64(1e-30) {
                 break; 
            }
            
            let beta = (rho / rho_old) * (alpha / omega);
            
            // p = r + beta * (p - omega * v)
            let size = x.size();
            {
                let p_data = p.storage_mut().data_mut();
                let r_data = r.storage().data();
                let v_data = v.storage().data();
                for k in 0..size {
                    p_data[k] = r_data[k] + beta * (p_data[k] - omega * v_data[k]);
                }
            }
            
            // v = A * p
            let ap = a.mul_dense(&p).expect("Sparse-Dense multiplication failed");
            v.assign(&ap).unwrap();
            
            let r_hat_v = r_hat.dot(&v);
             if r_hat_v.abs() < T::from_f64(1e-30) {
                 break; 
            }
            
            alpha = rho / r_hat_v;
            
            // s = r - alpha * v
            {
                let s_data = s.storage_mut().data_mut();
                let r_data = r.storage().data();
                let v_data = v.storage().data();
                for k in 0..size {
                    s_data[k] = r_data[k] - alpha * v_data[k];
                }
            }
            
            let s_norm_sq = s.squared_norm();
            if s_norm_sq < tol_sq {
                // x = x + alpha * p
                let x_data = x.storage_mut().data_mut();
                let p_data = p.storage().data();
                for k in 0..size {
                    x_data[k] += alpha * p_data[k];
                }
                *self.error.borrow_mut() = s_norm_sq.sqrt();
                break;
            }
            
            // t = A * s
             let as_ = a.mul_dense(&s).expect("Sparse-Dense multiplication failed");
             t.assign(&as_).unwrap();
            
            // omega = (t . s) / (t . t)
            let t_t = t.squared_norm();
            if t_t.abs() < T::from_f64(1e-30) {
                 break; 
            }
            omega = t.dot(&s) / t_t;
            
            // x = x + alpha * p + omega * s
            // r = s - omega * t
            {
                let x_data = x.storage_mut().data_mut();
                let p_data = p.storage().data();
                let s_data = s.storage().data();
                
                let r_data = r.storage_mut().data_mut();
                let t_data = t.storage().data();
                
                for k in 0..size {
                    x_data[k] += alpha * p_data[k] + omega * s_data[k];
                    r_data[k] = s_data[k] - omega * t_data[k];
                }
            }
            
            error_sq = r.squared_norm();
            *self.error.borrow_mut() = error_sq.sqrt();
            
            if error_sq < tol_sq {
                break;
            }
            
            if omega.abs() < T::from_f64(1e-30) {
                 break; 
            }
            
            rho_old = rho;
        }
        
        x
    }
    
    fn iterations(&self) -> usize {
        *self.iterations.borrow()
    }
    
    fn error(&self) -> T {
        *self.error.borrow()
    }
}
