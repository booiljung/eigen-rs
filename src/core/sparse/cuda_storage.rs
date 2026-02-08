//! Storage for sparse matrices on CUDA devices.

use crate::core::scalar::Scalar;
use std::ptr::NonNull;

/// Storage for a sparse matrix in CSR format on a CUDA device.
pub struct CudaSparseStorage<T: Scalar> {
    pub values: NonNull<T>,
    pub col_indices: NonNull<i32>,
    pub row_offsets: NonNull<i32>,
    pub nnz: usize,
    pub rows: usize,
    pub cols: usize,
}

impl<T: Scalar> CudaSparseStorage<T> {
    pub fn new(rows: usize, cols: usize, nnz: usize) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_sys::cudart::{cudaMalloc, cudaError_t};
            
            let mut values_ptr: *mut T = std::ptr::null_mut();
            let mut col_ptr: *mut i32 = std::ptr::null_mut();
            let mut row_ptr: *mut i32 = std::ptr::null_mut();

            let res = cudaMalloc(&mut values_ptr as *mut *mut T as *mut *mut std::ffi::c_void, nnz * std::mem::size_of::<T>());
            if res != cudaError_t::Success {
                return Err("Failed to allocate values on CUDA".to_string());
            }

            let res = cudaMalloc(&mut col_ptr as *mut *mut i32 as *mut *mut std::ffi::c_void, nnz * std::mem::size_of::<i32>());
            if res != cudaError_t::Success {
                return Err("Failed to allocate col_indices on CUDA".to_string());
            }

            let res = cudaMalloc(&mut row_ptr as *mut *mut i32 as *mut *mut std::ffi::c_void, (rows + 1) * std::mem::size_of::<i32>());
            if res != cudaError_t::Success {
                return Err("Failed to allocate row_offsets on CUDA".to_string());
            }
            
            Ok(Self {
                values: NonNull::new(values_ptr).unwrap(),
                col_indices: NonNull::new(col_ptr).unwrap(),
                row_offsets: NonNull::new(row_ptr).unwrap(),
                nnz,
                rows,
                cols,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (rows, cols, nnz);
            Err("CUDA feature not enabled".to_string())
        }
    }

    #[cfg(feature = "cuda")]
    pub fn copy_from_host(&mut self, values: &[T], col_indices: &[i32], row_offsets: &[i32]) -> Result<(), String> {
        use cuda_sys::cudart::{cudaMemcpy, cudaError_t};
        unsafe {
            let res = cudaMemcpy(self.values.as_ptr() as *mut _, values.as_ptr() as *const _, values.len() * std::mem::size_of::<T>(), 1);
            if res != cudaError_t::Success { return Err("H2D Copy failed (values)".to_string()); }

            let res = cudaMemcpy(self.col_indices.as_ptr() as *mut _, col_indices.as_ptr() as *const _, col_indices.len() * std::mem::size_of::<i32>(), 1);
            if res != cudaError_t::Success { return Err("H2D Copy failed (col_indices)".to_string()); }

            let res = cudaMemcpy(self.row_offsets.as_ptr() as *mut _, row_offsets.as_ptr() as *const _, row_offsets.len() * std::mem::size_of::<i32>(), 1);
            if res != cudaError_t::Success { return Err("H2D Copy failed (row_offsets)".to_string()); }
        }
        Ok(())
    }
}

impl<T: Scalar> Drop for CudaSparseStorage<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_sys::cudart::cudaFree;
            let _ = cudaFree(self.values.as_ptr() as *mut _);
            let _ = cudaFree(self.col_indices.as_ptr() as *mut _);
            let _ = cudaFree(self.row_offsets.as_ptr() as *mut _);
        }
    }
}
