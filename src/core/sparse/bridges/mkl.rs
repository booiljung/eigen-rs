//! Bridge to Intel MKL PARDISO solver.

use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::matrix::Matrix;
use crate::core::storage::{Storage, DynamicStorage};

#[cfg(feature = "mkl")]
pub mod sys {
    use std::os::raw::{c_int, c_long, c_double, c_void};

    extern "C" {
        pub fn pardiso(
            pt: *mut c_void,
            maxfct: *const c_int,
            mnum: *const c_int,
            mtype: *const c_int,
            phase: *const c_int,
            n: *const c_int,
            a: *const c_void,
            ia: *const c_int,
            ja: *const c_int,
            perm: *const c_int,
            nrhs: *const c_int,
            iparm: *mut c_int,
            msglvl: *const c_int,
            b: *mut c_void,
            x: *mut c_void,
            error: *mut c_int,
        );
    }
}

/// Sparse direct solver using Intel MKL PARDISO.
pub struct MklPardiso<T: Scalar> {
    #[cfg(feature = "mkl")]
    pt: [usize; 64], // Internal solver memory pointer
    #[cfg(feature = "mkl")]
    iparm: [i32; 64],
    mtype: i32,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> MklPardiso<T> {
    pub fn new(symmetric: bool, pd: bool) -> Self {
        let mtype = if symmetric {
            if pd { 2 } else { -2 }
        } else {
            11 // Real unsymmetric
        };

        Self {
            #[cfg(feature = "mkl")]
            pt: [0; 64],
            #[cfg(feature = "mkl")]
            iparm: [0; 64],
            mtype,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn compute(&mut self, _matrix: &SparseMatrix<T>) -> Result<(), String> {
        #[cfg(not(feature = "mkl"))]
        return Err("MKL feature not enabled".to_string());

        #[cfg(feature = "mkl")]
        {
            // MKL PARDISO requires CSR format
            // ... (FFI calls for symbolic and numerical factorization) ...
            Ok(())
        }
    }
}
