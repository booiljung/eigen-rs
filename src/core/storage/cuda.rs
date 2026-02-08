use crate::core::storage::Storage;
use crate::core::scalar::Scalar;
use std::ptr::NonNull;

/// Storage backed by CUDA device memory.
pub struct CudaStorage<T: Scalar> {
    data: NonNull<T>,
    rows: usize,
    cols: usize,
}

unsafe impl<T: Scalar> Send for CudaStorage<T> {}
unsafe impl<T: Scalar> Sync for CudaStorage<T> {}

impl<T: Scalar> CudaStorage<T> {
    pub fn new(rows: usize, cols: usize) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            let size = rows * cols;
            let mut ptr: *mut T = std::ptr::null_mut();
            
            unsafe {
                use cuda_sys::cudart::{cudaMalloc, cudaError_t};
                let res = cudaMalloc(&mut ptr as *mut *mut T as *mut *mut std::ffi::c_void, size * std::mem::size_of::<T>());
                if res != cudaError_t::Success {
                    return Err(format!("CUDA malloc failed with error code: {:?}", res));
                }
            }
    
            Ok(Self {
                data: NonNull::new(ptr).ok_or("Failed to create NonNull from CUDA pointer")?,
                rows,
                cols,
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (rows, cols);
            Err("CUDA feature not enabled".to_string())
        }
    }

    /// Copy data from CPU (Host) to GPU (Device).
    pub fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), String> {
        if host_data.len() != self.rows * self.cols {
            return Err("Dimension mismatch in copy_from_host".to_string());
        }

        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_sys::cudart::{cudaMemcpy, cudaError_t};
            // cudaMemcpyHostToDevice is usually 1
            let res = cudaMemcpy(
                self.data.as_ptr() as *mut std::ffi::c_void,
                host_data.as_ptr() as *const std::ffi::c_void,
                host_data.len() * std::mem::size_of::<T>(),
                1 // cudaMemcpyHostToDevice
            );
            if res != cudaError_t::Success {
                return Err(format!("CUDA memcpy H2D failed: {:?}", res));
            }
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = host_data;
            Err("CUDA not enabled".to_string())
        }
    }

    /// Copy data from GPU (Device) to CPU (Host).
    pub fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), String> {
        if host_data.len() != self.rows * self.cols {
            return Err("Dimension mismatch in copy_to_host".to_string());
        }

        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_sys::cudart::{cudaMemcpy, cudaError_t};
            // cudaMemcpyDeviceToHost is usually 2
            let res = cudaMemcpy(
                host_data.as_mut_ptr() as *mut std::ffi::c_void,
                self.data.as_ptr() as *const std::ffi::c_void,
                host_data.len() * std::mem::size_of::<T>(),
                2 // cudaMemcpyDeviceToHost
            );
            if res != cudaError_t::Success {
                return Err(format!("CUDA memcpy D2H failed: {:?}", res));
            }
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = host_data;
            Err("CUDA not enabled".to_string())
        }
    }
}

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    fn data(&self) -> &[T] { &[] }
    fn data_mut(&mut self) -> &mut [T] { &mut [] }

    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    
    fn get_ptr(&self, row: usize, col: usize) -> *const T {
        unsafe { self.data.as_ptr().add(col * self.rows + row) }
    }

    fn as_cuda_storage(&self) -> Option<&CudaStorage<T>> {
        Some(self)
    }
}

impl<T: Scalar> Drop for CudaStorage<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            use cuda_sys::cudart::cudaFree;
            let _ = cudaFree(self.data.as_ptr() as *mut std::ffi::c_void);
        }
    }
}
