use crate::core::scalar::Scalar;
use crate::core::storage::CudaStorage;
use crate::core::storage::Storage;

#[cfg(feature = "cuda")]
pub mod sys {
    pub use crate::core::tensor::device::cublas::*;
}

#[cfg(not(feature = "cuda"))]
pub mod sys {
    #[allow(non_camel_case_types)]
    pub type cublasHandle_t = ();
    #[allow(non_camel_case_types)]
    pub type cublasStatus_t = i32;
    pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
}

pub struct CublasHandle {
    #[cfg(feature = "cuda")]
    handle: sys::cublasHandle_t,
    #[cfg(not(feature = "cuda"))]
    _handle: sys::cublasHandle_t,
}

impl CublasHandle {
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            let mut handle = std::ptr::null_mut();
            unsafe {
                let api = sys::get_cublas().ok_or("Failed to load cuBLAS API")?;
                let res = (api.cublasCreate_v2)(&mut handle);
                if res != sys::CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasCreate failed: {:?}", res));
                }
            }
            Ok(Self { handle })
        }
        #[cfg(not(feature = "cuda"))]
        Err("CUDA not enabled".to_string())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            if let Some(api) = sys::get_cublas() {
                (api.cublasDestroy_v2)(self.handle);
            }
        }
    }
}

/// Perform C = alpha * A * B + beta * C
/// A: m x k
/// B: k x n
/// C: m x n
#[allow(clippy::too_many_arguments)]
pub fn gemm_cublas<T: Scalar>(
    handle: &CublasHandle,
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    beta: T,
    c: &mut CudaStorage<T>,
) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    unsafe {
        let op_a = if trans_a {
            sys::CUBLAS_OP_T
        } else {
            sys::CUBLAS_OP_N
        };
        let op_b = if trans_b {
            sys::CUBLAS_OP_T
        } else {
            sys::CUBLAS_OP_N
        };

        // LDA/LDB/LDC = Leading Dimension.
        // If Column Major:
        // A is m x k. If trans_a=N, LDA=m. If trans_a=T, LDA=k.
        // Wait, cuBLAS expects Column Major by default.
        // eigen-rs uses Column Major.
        // If NoTrans, A is m x k, stored as m x k. lda = m.

        let _lda = if trans_a { k } else { m } as i32; // Assuming packed
                                                       // Actually for standard storage: lda = rows() always? No, stride.
                                                       // CudaStorage packs contiguously.
        let lda = Storage::<T>::rows(a) as i32;
        let ldb = Storage::<T>::rows(b) as i32;
        let ldc = Storage::<T>::rows(c) as i32;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let api = sys::get_cublas().ok_or("Failed to load cuBLAS API")?;
            let alpha_f32 = *(&alpha as *const T as *const f32);
            let beta_f32 = *(&beta as *const T as *const f32);

            let res = (api.cublasSgemm_v2)(
                handle.handle,
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha_f32,
                Storage::<T>::get_ptr(a, 0, 0) as *const f32,
                lda,
                Storage::<T>::get_ptr(b, 0, 0) as *const f32,
                ldb,
                &beta_f32,
                Storage::<T>::get_ptr(c, 0, 0) as *mut f32,
                ldc,
            );

            if res != sys::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSgemm failed: {:?}", res));
            }
            Ok(())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let api = sys::get_cublas().ok_or("Failed to load cuBLAS API")?;
            let alpha_f64 = *(&alpha as *const T as *const f64);
            let beta_f64 = *(&beta as *const T as *const f64);

            let res = (api.cublasDgemm_v2)(
                handle.handle,
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha_f64,
                Storage::<T>::get_ptr(a, 0, 0) as *const f64,
                lda,
                Storage::<T>::get_ptr(b, 0, 0) as *const f64,
                ldb,
                &beta_f64,
                Storage::<T>::get_ptr(c, 0, 0) as *mut f64,
                ldc,
            );

            if res != sys::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasDgemm failed: {:?}", res));
            }
            Ok(())
        } else {
            Err("Unsupported type for cuBLAS GEMM".to_string())
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (handle, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
        Err("CUDA not enabled".to_string())
    }
}
