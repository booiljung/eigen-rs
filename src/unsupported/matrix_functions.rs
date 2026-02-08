
//! Unsupported / Experimental Matrix Functions.
//! Implementation of Matrix Exponential and Logarithm.

use crate::core::matrix::{Matrix, DynamicStorage};
use crate::core::storage::Storage;
use crate::core::scalar::Scalar;
use crate::core::xpr::MatrixXpr;
use crate::core::decompositions::lu::PartialPivLU; // Needed for solving D^-1 N

/// Matrix Exponential function e^A.
/// 
/// Uses scaling and squaring method with Pade approximation of order [6/6].
/// Reference: "The Scaling and Squaring Method for the Matrix Exponential Revisited", Higham 2005.
pub struct MatrixExponential<'a, T, S> 
where 
    T: Scalar,
    S: Storage<T>
{
    matrix: &'a Matrix<T, S>,
}

impl<'a, T, S> MatrixExponential<'a, T, S> 
where 
    T: Scalar + Copy + PartialOrd + StaticFloatConsts + 'static,
    S: Storage<T> + 'static
{
    pub fn new(matrix: &'a Matrix<T, S>) -> Self {
        Self { matrix }
    }

    /// Compute the matrix exponential.
    pub fn compute(&self) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.matrix.rows();
        let cols = self.matrix.cols();
        if rows != cols {
            return Err("Matrix must be square".to_string());
        }
        
        // 1. Calculate infinity norm (max row sum of abs values)
        let mut norm_inf = T::default();
        for i in 0..rows {
            let mut row_sum = T::default();
            for j in 0..cols {
                row_sum += self.matrix.eval(i, j).abs();
            }
            if row_sum > norm_inf {
                norm_inf = row_sum;
            }
        }
        
        // 2. Scaling: Find s such that ||A||_inf / 2^s <= 0.5
        let mut s = 0;
        let mut val = norm_inf;
        let limit = T::from_f64(0.5);
        
        // Safety against Infinite norm loop
        if val > limit && val < T::from_f64(1e30) { 
             while val > limit {
                 val *= T::from_f64(0.5);
                 s += 1;
                 if s > 100 { break; } 
             }
        } else if val > limit {
             s = 20; 
        }
        
        // Copy A and scale
        let mut a = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        let scale_factor = T::from_f64(1.0) / T::from_f64(2.0).powf(T::from_f64(s as f64));
        
        for i in 0..rows {
            for j in 0..cols {
                *a.get_mut(i, j).unwrap() = self.matrix.eval(i, j) * scale_factor;
            }
        }
        
        // 3. Pade Approximation [6/6]
        // let c0 = T::from_f64(1.0); // c0 is 1.0, implicit in i_mat initialization
        let c1 = T::from_f64(0.5);
        let c2 = T::from_f64(5.0 / 44.0);
        let c3 = T::from_f64(1.0 / 66.0);
        let c4 = T::from_f64(1.0 / 792.0);
        let c5 = T::from_f64(1.0 / 15840.0);
        let c6 = T::from_f64(1.0 / 665280.0);
        
        // Calculate powers
        let a2 = self.mul_matrix(&a, &a);
        let a4 = self.mul_matrix(&a2, &a2);
        let a6 = self.mul_matrix(&a4, &a2);
        
        let i_mat = identity(rows, cols);
        
        // E = c0*I + c2*A2 + c4*A4 + c6*A6
        let mut e = i_mat.clone();
        // Since c0=1, e is already I.
        // But if c0 != 1 (unlikely for exp), we would need scaling.
        // Actually Pade N_6 has term 1.0 * I.
        // So e starts as I.
        
        self.add_scaled(&mut e, &a2, c2);
        self.add_scaled(&mut e, &a4, c4);
        self.add_scaled(&mut e, &a6, c6);
        
        // O_poly = c1*I + c3*A2 + c5*A4
        let mut o_poly = i_mat.clone(); 
        
        for r in 0..rows { for c in 0..cols { *o_poly.get_mut(r, c).unwrap() = T::default(); } }
        for i in 0..rows { *o_poly.get_mut(i, i).unwrap() = c1; }
        
        self.add_scaled(&mut o_poly, &a2, c3);
        self.add_scaled(&mut o_poly, &a4, c5);
        
        // O = A * O_poly
        let o = self.mul_matrix(&a, &o_poly);
        
        // N = E + O
        // D = E - O
        let mut n = e.clone();
        for r in 0..rows { for c in 0..cols { 
            let val_o = o.eval(r, c);
            let val_e = e.eval(r, c);
            *n.get_mut(r, c).unwrap() = val_e + val_o;
        }}
        
        let mut d = e; 
        for r in 0..rows { for c in 0..cols { 
            let val_o = o.eval(r, c);
            let val_e = d.eval(r, c);
            *d.get_mut(r, c).unwrap() = val_e - val_o;
        }}
        
        // Solve D * R = N
        let lu = PartialPivLU::new(&d)?;
        let mut r = lu.solve(&n)?;
        
        // 4. Squaring
        for _ in 0..s {
            r = self.mul_matrix(&r, &r);
        }
        
        Ok(r)
    }
    
    // Helpers
    fn add_scaled(&self, dest: &mut Matrix<T, DynamicStorage<T>>, src: &Matrix<T, DynamicStorage<T>>, scalar: T) {
        let rows = dest.rows();
        let cols = dest.cols();
        for i in 0..rows {
            for j in 0..cols {
                let val = (*dest).eval(i, j) + src.eval(i, j) * scalar;
                 *dest.get_mut(i, j).unwrap() = val;
            }
        }
    }
    
    fn mul_matrix(&self, lhs: &Matrix<T, DynamicStorage<T>>, rhs: &Matrix<T, DynamicStorage<T>>) -> Matrix<T, DynamicStorage<T>> {
        let prod = lhs * rhs;
        let mut res = Matrix::<T, DynamicStorage<T>>::new_dynamic(lhs.rows(), rhs.cols()).unwrap();
        res.assign(&prod).unwrap();
        res
    }
}

/// Matrix Logarithm function log(A).
pub struct MatrixLogarithm<'a, T, S> 
where 
    T: Scalar,
    S: Storage<T>
{
    matrix: &'a Matrix<T, S>,
}

impl<'a, T, S> MatrixLogarithm<'a, T, S> 
where 
    T: Scalar + Copy + PartialOrd + StaticFloatConsts + 'static,
    S: Storage<T> + 'static
{
    pub fn new(matrix: &'a Matrix<T, S>) -> Self {
        Self { matrix }
    }

    pub fn compute(&self) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.matrix.rows();
        let cols = self.matrix.cols();
        if rows != cols {
            return Err("Matrix must be square".to_string());
        }
        
        let mut a = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        // Copy
        for i in 0..rows { for j in 0..cols { *a.get_mut(i, j).unwrap() = self.matrix.eval(i, j); }}
        
        let i_mat = identity(rows, cols);
        
        let mut k = 0;
        let p_val = T::from_f64(0.3);
        
        // Max iterations for sqrt reduction
        for _ in 0..10 { 
            let diff = self.sub_matrix(&a, &i_mat);
            let norm = self.norm_inf(&diff);
            
            if norm <= p_val {
                break;
            }
            
            // Denman-Beavers iteration for square root
            let mut y = a.clone();
            let mut z = i_mat.clone();
            let half = T::from_f64(0.5);
            
            for _ in 0..10 { 
                 let lu_y = PartialPivLU::new(&y)?;
                 let inv_y = lu_y.solve(&i_mat)?;
                 
                 let lu_z = PartialPivLU::new(&z)?;
                 let inv_z = lu_z.solve(&i_mat)?;
                 
                 let ny = self.add_matrix(&y, &inv_z); // y + inv_z
                 let mut ny_scaled = ny;
                 self.scale_matrix(&mut ny_scaled, half);
                 
                 let nz = self.add_matrix(&z, &inv_y);
                 let mut nz_scaled = nz;
                 self.scale_matrix(&mut nz_scaled, half);
                 
                 y = ny_scaled;
                 z = nz_scaled;
            }
            a = y;
            k += 1;
        }
        
        // 2. Pade / Taylor Approx for Log(A)
        // X = A - I
        let x_mat = self.sub_matrix(&a, &i_mat);
        let x2 = self.mul_matrix(&x_mat, &x_mat);
        let x3 = self.mul_matrix(&x2, &x_mat);
        let x4 = self.mul_matrix(&x3, &x_mat);
        
        let mut l = x_mat.clone(); // term 1: X
        
        // - X^2/2
        self.add_scaled(&mut l, &x2, T::from_f64(-0.5));
        // + X^3/3
        self.add_scaled(&mut l, &x3, T::from_f64(1.0/3.0));
        // - X^4/4
        self.add_scaled(&mut l, &x4, T::from_f64(-0.25));
        
        // 3. Rescale: result * 2^k
        let scale = T::from_f64(2.0).powf(T::from_f64(k as f64));
        self.scale_matrix(&mut l, scale);
        
        Ok(l)
    }

    // Helpers (for Logarithm)
    fn norm_inf(&self, m: &Matrix<T, DynamicStorage<T>>) -> T {
        let mut norm = T::default();
        for i in 0..m.rows() {
            let mut sum = T::default();
            for j in 0..m.cols() { sum += m.eval(i, j).abs(); }
            if sum > norm { norm = sum; }
        }
        norm
    }
    
    fn sub_matrix(&self, a: &Matrix<T, DynamicStorage<T>>, b: &Matrix<T, DynamicStorage<T>>) -> Matrix<T, DynamicStorage<T>> {
        let mut r = a.clone();
        for i in 0..a.rows() { for j in 0..a.cols() {
            let val = a.eval(i, j) - b.eval(i, j);
            *r.get_mut(i, j).unwrap() = val;
        }}
        r
    }

    fn add_matrix(&self, a: &Matrix<T, DynamicStorage<T>>, b: &Matrix<T, DynamicStorage<T>>) -> Matrix<T, DynamicStorage<T>> {
        let mut r = a.clone();
        for i in 0..a.rows() { for j in 0..a.cols() {
            let val = a.eval(i, j) + b.eval(i, j);
            *r.get_mut(i, j).unwrap() = val;
        }}
        r
    }
    
    fn mul_matrix(&self, lhs: &Matrix<T, DynamicStorage<T>>, rhs: &Matrix<T, DynamicStorage<T>>) -> Matrix<T, DynamicStorage<T>> {
        let prod = lhs * rhs;
        let mut res = Matrix::<T, DynamicStorage<T>>::new_dynamic(lhs.rows(), rhs.cols()).unwrap();
        res.assign(&prod).unwrap();
        res
    }
    
    fn add_scaled(&self, dest: &mut Matrix<T, DynamicStorage<T>>, src: &Matrix<T, DynamicStorage<T>>, scalar: T) {
         for i in 0..dest.rows() { for j in 0..dest.cols() {
             let val = (*dest).eval(i, j) + src.eval(i, j) * scalar;
             *dest.get_mut(i, j).unwrap() = val;
         }}
    }
    
    fn scale_matrix(&self, dest: &mut Matrix<T, DynamicStorage<T>>, scalar: T) {
         for i in 0..dest.rows() { for j in 0..dest.cols() {
             let val = (*dest).eval(i, j) * scalar;
             *dest.get_mut(i, j).unwrap() = val;
         }}
    }
}

// Helper trait to avoid bounds mess
#[doc(hidden)]
pub trait StaticFloatConsts {
}
impl<T: Scalar> StaticFloatConsts for T {}

// Module-level helper for Identity
fn identity<T: Scalar>(rows: usize, cols: usize) -> Matrix<T, DynamicStorage<T>> {
    let mut m = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).expect("Failed to allocate identity matrix");
    for i in 0..rows {
        for j in 0..cols {
            let val = if i == j { T::from_usize(1) } else { T::from_usize(0) };
            *m.get_mut(i, j).unwrap() = val;
        }
    }
    m
}
