//! LAPACK bridges for eigen-rs.
//! Provides high-performance LU and QR decompositions.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

#[cfg(feature = "lapack")]
extern crate lapack_sys;

/// LU decomposition using LAPACK.
pub struct LapackLU<T: Scalar> {
    rows: usize,
    cols: usize,
    lu: Matrix<T, DynamicStorage<T>>,
    #[allow(dead_code)]
    ipiv: Vec<i32>,
    is_initialized: bool,
}

impl<T: Scalar> LapackLU<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            lu: Matrix::new_dynamic(rows, cols).unwrap(),
            ipiv: vec![0; std::cmp::min(rows, cols)],
            is_initialized: false,
        }
    }

    pub fn compute<S: Storage<T>>(&mut self, matrix: &Matrix<T, S>) -> Result<(), String> {
        self.rows = matrix.rows();
        self.cols = matrix.cols();
        self.lu.assign(matrix)?;

        #[cfg(feature = "lapack")]
        unsafe {
            let m = self.rows as i32;
            let n = self.cols as i32;
            let lda = m;
            self.ipiv = vec![0; std::cmp::min(self.rows, self.cols) as usize];
            let mut info = 0i32;

            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                lapack_sys::sgetrf_(
                    &m,
                    &n,
                    self.lu.storage_mut().data_mut().as_mut_ptr() as *mut f32,
                    &lda,
                    self.ipiv.as_mut_ptr(),
                    &mut info,
                );
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                lapack_sys::dgetrf_(
                    &m,
                    &n,
                    self.lu.storage_mut().data_mut().as_mut_ptr() as *mut f64,
                    &lda,
                    self.ipiv.as_mut_ptr(),
                    &mut info,
                );
            } else {
                return Err("LAPACK only supports f32 and f64".to_string());
            }

            if info < 0 {
                return Err(format!(
                    "LAPACK getrf: argument {} had illegal value",
                    -info
                ));
            } else if info > 0 {
                return Err(format!(
                    "LAPACK getrf: matrix is singular, U({}, {}) is exactly zero",
                    info, info
                ));
            }

            self.is_initialized = true;
            Ok(())
        }

        #[cfg(not(feature = "lapack"))]
        {
            Err("LAPACK feature not enabled".to_string())
        }
    }

    pub fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_initialized {
            return Err("LapackLU not initialized".to_string());
        }
        if b.rows() != self.rows {
            return Err("Dimension mismatch in solve".to_string());
        }

        #[cfg(feature = "lapack")]
        unsafe {
            let mut x = Matrix::new_dynamic(b.rows(), b.cols())?;
            x.assign(b)?;

            let n = self.rows as i32;
            let nrhs = b.cols() as i32;
            let lda = n;
            let ldb = n;
            let mut info = 0i32;
            let trans = 'N' as i8;

            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                lapack_sys::sgetrs_(
                    &trans,
                    &n,
                    &nrhs,
                    self.lu.storage().data().as_ptr() as *const f32,
                    &lda,
                    self.ipiv.as_ptr(),
                    x.storage_mut().data_mut().as_mut_ptr() as *mut f32,
                    &ldb,
                    &mut info,
                );
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                lapack_sys::dgetrs_(
                    &trans,
                    &n,
                    &nrhs,
                    self.lu.storage().data().as_ptr() as *const f64,
                    &lda,
                    self.ipiv.as_ptr(),
                    x.storage_mut().data_mut().as_mut_ptr() as *mut f64,
                    &ldb,
                    &mut info,
                );
            }

            if info != 0 {
                return Err(format!("LAPACK getrs failed with info {}", info));
            }

            Ok(x)
        }

        #[cfg(not(feature = "lapack"))]
        {
            Err("LAPACK feature not enabled".to_string())
        }
    }
}

/// QR decomposition using LAPACK.
pub struct LapackQR<T: Scalar> {
    rows: usize,
    cols: usize,
    qr: Matrix<T, DynamicStorage<T>>,
    #[allow(dead_code)]
    tau: Vec<T>,
    is_initialized: bool,
}

impl<T: Scalar> LapackQR<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            qr: Matrix::new_dynamic(rows, cols).unwrap(),
            tau: Vec::new(),
            is_initialized: false,
        }
    }

    pub fn compute<S: Storage<T>>(&mut self, matrix: &Matrix<T, S>) -> Result<(), String> {
        self.rows = matrix.rows();
        self.cols = matrix.cols();
        self.qr.assign(matrix)?;

        #[cfg(feature = "lapack")]
        unsafe {
            let m = self.rows as i32;
            let n = self.cols as i32;
            let lda = m;
            self.tau = vec![T::default(); std::cmp::min(self.rows, self.cols)];
            let mut info = 0i32;

            let mut work = [T::default(); 1];
            let mut lwork = -1i32;

            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                // Workspace query
                lapack_sys::sgeqrf_(
                    &m,
                    &n,
                    std::ptr::null_mut(),
                    &lda,
                    std::ptr::null_mut(),
                    work.as_mut_ptr() as *mut f32,
                    &lwork,
                    &mut info,
                );
                lwork = work[0].to_f64() as i32;
                let mut vwork = vec![0.0f32; lwork as usize];
                lapack_sys::sgeqrf_(
                    &m,
                    &n,
                    self.qr.storage_mut().data_mut().as_mut_ptr() as *mut f32,
                    &lda,
                    self.tau.as_mut_ptr() as *mut f32,
                    vwork.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                lapack_sys::dgeqrf_(
                    &m,
                    &n,
                    std::ptr::null_mut(),
                    &lda,
                    std::ptr::null_mut(),
                    work.as_mut_ptr() as *mut f64,
                    &lwork,
                    &mut info,
                );
                lwork = work[0].to_f64() as i32;
                let mut vwork = vec![0.0f64; lwork as usize];
                lapack_sys::dgeqrf_(
                    &m,
                    &n,
                    self.qr.storage_mut().data_mut().as_mut_ptr() as *mut f64,
                    &lda,
                    self.tau.as_mut_ptr() as *mut f64,
                    vwork.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );
            } else {
                return Err("LAPACK only supports f32 and f64".to_string());
            }

            if info != 0 {
                return Err(format!("LAPACK geqrf failed with info {}", info));
            }

            self.is_initialized = true;
            Ok(())
        }

        #[cfg(not(feature = "lapack"))]
        {
            Err("LAPACK feature not enabled".to_string())
        }
    }

    pub fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_initialized {
            return Err("LapackQR not initialized".to_string());
        }

        #[cfg(feature = "lapack")]
        unsafe {
            let m = self.rows as i32;
            let n = self.cols as i32;
            let nrhs = b.cols() as i32;
            let lda = m;
            let ldb = m;
            let k = std::cmp::min(m, n);

            let mut b_copy = Matrix::new_dynamic(b.rows(), b.cols())?;
            b_copy.assign(b)?;

            let mut info = 0i32;
            let side = 'L' as i8;
            let trans = 'T' as i8;

            let mut work = [T::default(); 1];
            let mut lwork = -1i32;

            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                lapack_sys::sormqr_(
                    &side,
                    &trans,
                    &m,
                    &nrhs,
                    &k,
                    self.qr.storage().data().as_ptr() as *const f32,
                    &lda,
                    self.tau.as_ptr() as *const f32,
                    b_copy.storage_mut().data_mut().as_mut_ptr() as *mut f32,
                    &ldb,
                    work.as_mut_ptr() as *mut f32,
                    &lwork,
                    &mut info,
                );
                lwork = work[0].to_f64() as i32;
                let mut vwork = vec![0.0f32; lwork as usize];
                lapack_sys::sormqr_(
                    &side,
                    &trans,
                    &m,
                    &nrhs,
                    &k,
                    self.qr.storage().data().as_ptr() as *const f32,
                    &lda,
                    self.tau.as_ptr() as *const f32,
                    b_copy.storage_mut().data_mut().as_mut_ptr() as *mut f32,
                    &ldb,
                    vwork.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );

                if info == 0 {
                    // Back substitution with R
                    let uplo = 'U' as i8;
                    let transa = 'N' as i8;
                    let diag = 'N' as i8;
                    lapack_sys::strtrs_(
                        &uplo,
                        &transa,
                        &diag,
                        &n,
                        &nrhs,
                        self.qr.storage().data().as_ptr() as *const f32,
                        &lda,
                        b_copy.storage_mut().data_mut().as_mut_ptr() as *mut f32,
                        &ldb,
                        &mut info,
                    );
                }
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                lapack_sys::dormqr_(
                    &side,
                    &trans,
                    &m,
                    &nrhs,
                    &k,
                    self.qr.storage().data().as_ptr() as *const f64,
                    &lda,
                    self.tau.as_ptr() as *const f64,
                    b_copy.storage_mut().data_mut().as_mut_ptr() as *mut f64,
                    &ldb,
                    work.as_mut_ptr() as *mut f64,
                    &lwork,
                    &mut info,
                );
                lwork = work[0].to_f64() as i32;
                let mut vwork = vec![0.0f64; lwork as usize];
                lapack_sys::dormqr_(
                    &side,
                    &trans,
                    &m,
                    &nrhs,
                    &k,
                    self.qr.storage().data().as_ptr() as *const f64,
                    &lda,
                    self.tau.as_ptr() as *const f64,
                    b_copy.storage_mut().data_mut().as_mut_ptr() as *mut f64,
                    &ldb,
                    vwork.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );

                if info == 0 {
                    let uplo = 'U' as i8;
                    let transa = 'N' as i8;
                    let diag = 'N' as i8;
                    lapack_sys::dtrtrs_(
                        &uplo,
                        &transa,
                        &diag,
                        &n,
                        &nrhs,
                        self.qr.storage().data().as_ptr() as *const f64,
                        &lda,
                        b_copy.storage_mut().data_mut().as_mut_ptr() as *mut f64,
                        &ldb,
                        &mut info,
                    );
                }
            }

            if info != 0 {
                return Err(format!("LAPACK QR solve failed with info {}", info));
            }

            // The result is in the first n rows of b_copy
            let mut x = Matrix::new_dynamic(self.cols, b.cols())?;
            for j in 0..b.cols() {
                for i in 0..self.cols {
                    *x.get_mut(i, j).unwrap() = *b_copy.get(i, j).unwrap();
                }
            }

            Ok(x)
        }

        #[cfg(not(feature = "lapack"))]
        {
            Err("LAPACK feature not enabled".to_string())
        }
    }
}
