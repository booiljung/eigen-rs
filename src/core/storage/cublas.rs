use crate::core::scalar::Scalar;
use crate::core::storage::CudaStorage;

#[cfg(feature = "cuda")]
pub mod sys {
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cublasContext {
        _unused: [u8; 0],
    }
    pub type cublasHandle_t = *mut cublasContext;

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum cublasStatus_t {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 3,
        InvalidValue = 7,
        ArchMismatch = 8,
        MappingError = 11,
        ExecutionFailed = 13,
        InternalError = 14,
        NotSupported = 15,
        LicenseError = 16,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum cublasOperation_t {
        N = 0,
        T = 1,
        C = 2,
    }

    unsafe extern "C" {
        pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
        pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;

        pub fn cublasSgemm_v2(
            handle: cublasHandle_t,
            transa: cublasOperation_t,
            transb: cublasOperation_t,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            A: *const f32,
            lda: i32,
            B: *const f32,
            ldb: i32,
            beta: *const f32,
            C: *mut f32,
            ldc: i32,
        ) -> cublasStatus_t;

        pub fn cublasDgemm_v2(
            handle: cublasHandle_t,
            transa: cublasOperation_t,
            transb: cublasOperation_t,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f64,
            A: *const f64,
            lda: i32,
            B: *const f64,
            ldb: i32,
            beta: *const f64,
            C: *mut f64,
            ldc: i32,
        ) -> cublasStatus_t;
    }

}

#[cfg(not(feature = "cuda"))]
pub mod sys {
    #[allow(non_camel_case_types)]
    pub type cublasHandle_t = ();
    #[allow(non_camel_case_types)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum cublasStatus_t {
        Success = 0,
    }
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
                let res = sys::cublasCreate_v2(&mut handle);
                if res != sys::cublasStatus_t::Success {
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
            sys::cublasDestroy_v2(self.handle);
        }
    }
}

/// Perform C = alpha * A * B + beta * C
/// A: m x k
/// B: k x n
/// C: m x n
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
            sys::cublasOperation_t::T
        } else {
            sys::cublasOperation_t::N
        };
        let op_b = if trans_b {
            sys::cublasOperation_t::T
        } else {
            sys::cublasOperation_t::N
        };

        // LDA/LDB/LDC = Leading Dimension.
        // If Column Major:
        // A is m x k. If trans_a=N, LDA=m. If trans_a=T, LDA=k.
        // Wait, cuBLAS expects Column Major by default.
        // eigen-rs uses Column Major.
        // If NoTrans, A is m x k, stored as m x k. lda = m.

        let lda = if trans_a { k } else { m } as i32; // Assuming packed
                                                      // Actually for standard storage: lda = rows() always? No, stride.
                                                      // CudaStorage packs contiguously.
        let lda = a.rows() as i32;
        let ldb = b.rows() as i32;
        let ldc = c.rows() as i32;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let alpha_f32 = *(&alpha as *const T as *const f32);
            let beta_f32 = *(&beta as *const T as *const f32);

            let res = sys::cublasSgemm_v2(
                handle.handle,
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha_f32,
                a.get_ptr(0, 0) as *const f32,
                lda,
                b.get_ptr(0, 0) as *const f32,
                ldb,
                &beta_f32,
                c.get_ptr(0, 0) as *mut f32,
                ldc,
            );

            if res != sys::cublasStatus_t::Success {
                return Err(format!("cublasSgemm failed: {:?}", res));
            }
            Ok(())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let alpha_f64 = *(&alpha as *const T as *const f64);
            let beta_f64 = *(&beta as *const T as *const f64);

            let res = sys::cublasDgemm_v2(
                handle.handle,
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha_f64,
                a.get_ptr(0, 0) as *const f64,
                lda,
                b.get_ptr(0, 0) as *const f64,
                ldb,
                &beta_f64,
                c.get_ptr(0, 0) as *mut f64,
                ldc,
            );

            if res != sys::cublasStatus_t::Success {
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
