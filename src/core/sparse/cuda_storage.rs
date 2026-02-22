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
        {
            let values_ptr =
                crate::core::tensor::device::cudart::cuda_malloc(nnz * std::mem::size_of::<T>())?
                    as *mut T;
            let col_ptr =
                crate::core::tensor::device::cudart::cuda_malloc(nnz * std::mem::size_of::<i32>())?
                    as *mut i32;
            let row_ptr = crate::core::tensor::device::cudart::cuda_malloc(
                (rows + 1) * std::mem::size_of::<i32>(),
            )? as *mut i32;

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
    pub fn copy_from_host(
        &mut self,
        values: &[T],
        col_indices: &[i32],
        row_offsets: &[i32],
    ) -> Result<(), String> {
        crate::core::tensor::device::cudart::cuda_memcpy_h2d(
            self.values.as_ptr() as *mut _,
            values.as_ptr() as *const _,
            values.len() * std::mem::size_of::<T>(),
        )?;

        crate::core::tensor::device::cudart::cuda_memcpy_h2d(
            self.col_indices.as_ptr() as *mut _,
            col_indices.as_ptr() as *const _,
            col_indices.len() * std::mem::size_of::<i32>(),
        )?;

        crate::core::tensor::device::cudart::cuda_memcpy_h2d(
            self.row_offsets.as_ptr() as *mut _,
            row_offsets.as_ptr() as *const _,
            row_offsets.len() * std::mem::size_of::<i32>(),
        )?;
        Ok(())
    }
}

impl<T: Scalar> Drop for CudaSparseStorage<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let _ = crate::core::tensor::device::cudart::cuda_free(self.values.as_ptr() as *mut _);
            let _ =
                crate::core::tensor::device::cudart::cuda_free(self.col_indices.as_ptr() as *mut _);
            let _ =
                crate::core::tensor::device::cudart::cuda_free(self.row_offsets.as_ptr() as *mut _);
        }
    }
}
