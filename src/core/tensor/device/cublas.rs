#![allow(non_snake_case)]

#[cfg(feature = "cuda")]
use libloading::Library;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[allow(non_camel_case_types)]
pub type cublasStatus_t = i32;
#[allow(non_camel_case_types)]
pub type cublasFillMode_t = i32;
pub const CUBLAS_FILL_MODE_LOWER: cublasFillMode_t = 0;
pub const CUBLAS_FILL_MODE_UPPER: cublasFillMode_t = 1;
#[allow(non_camel_case_types)]
pub type cublasOperation_t = i32;
#[allow(non_camel_case_types)]
pub enum cublasHandle {}
#[allow(non_camel_case_types)]
pub type cublasHandle_t = *mut cublasHandle;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;

#[cfg(feature = "cuda")]
pub struct CublasApi {
    _lib: Library,
    pub cublasCreate_v2: unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t,
    pub cublasDestroy_v2: unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t,
    pub cublasSgemm_v2: unsafe extern "C" fn(
        cublasHandle_t,
        cublasOperation_t,
        cublasOperation_t,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
    ) -> cublasStatus_t,
    pub cublasDgemm_v2: unsafe extern "C" fn(
        cublasHandle_t,
        cublasOperation_t,
        cublasOperation_t,
        i32,
        i32,
        i32,
        *const f64,
        *const f64,
        i32,
        *const f64,
        i32,
        *const f64,
        *mut f64,
        i32,
    ) -> cublasStatus_t,
    pub cublasSaxpy_v2: unsafe extern "C" fn(
        cublasHandle_t,
        i32,
        *const f32,
        *const f32,
        i32,
        *mut f32,
        i32,
    ) -> cublasStatus_t,
    pub cublasDaxpy_v2: unsafe extern "C" fn(
        cublasHandle_t,
        i32,
        *const f64,
        *const f64,
        i32,
        *mut f64,
        i32,
    ) -> cublasStatus_t,
    pub cublasSscal_v2:
        unsafe extern "C" fn(cublasHandle_t, i32, *const f32, *mut f32, i32) -> cublasStatus_t,
    pub cublasDscal_v2:
        unsafe extern "C" fn(cublasHandle_t, i32, *const f64, *mut f64, i32) -> cublasStatus_t,
    pub cublasSdot_v2: unsafe extern "C" fn(
        cublasHandle_t,
        i32,
        *const f32,
        i32,
        *const f32,
        i32,
        *mut f32,
    ) -> cublasStatus_t,
    pub cublasDdot_v2: unsafe extern "C" fn(
        cublasHandle_t,
        i32,
        *const f64,
        i32,
        *const f64,
        i32,
        *mut f64,
    ) -> cublasStatus_t,
    pub cublasSnrm2_v2:
        unsafe extern "C" fn(cublasHandle_t, i32, *const f32, i32, *mut f32) -> cublasStatus_t,
    pub cublasDnrm2_v2:
        unsafe extern "C" fn(cublasHandle_t, i32, *const f64, i32, *mut f64) -> cublasStatus_t,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CublasApi {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CublasApi {}

#[cfg(feature = "cuda")]
static CUBLAS_API: OnceLock<Option<Arc<CublasApi>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn get_cublas() -> Option<Arc<CublasApi>> {
    CUBLAS_API
        .get_or_init(|| {
            unsafe {
                // macOS / Windows variants (e.g. cublas.dll or libcublas.dylib) could be supported here,
                // but for now linux .so is targeted.
                let lib = match Library::new("libcublas.so") {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("eigen-rs [cuda]: Failed to load libcublas.so: {}", e);
                        return None;
                    }
                };

                let get_sym = |name: &[u8]| -> *const () {
                    match lib.get::<*const ()>(name) {
                        Ok(sym) => *sym,
                        Err(_) => std::ptr::null(),
                    }
                };

                let cublasCreate_v2_ptr = get_sym(b"cublasCreate_v2\0");
                let cublasDestroy_v2_ptr = get_sym(b"cublasDestroy_v2\0");
                let cublasSgemm_v2_ptr = get_sym(b"cublasSgemm_v2\0");
                let cublasDgemm_v2_ptr = get_sym(b"cublasDgemm_v2\0");
                let cublasSaxpy_v2_ptr = get_sym(b"cublasSaxpy_v2\0");
                let cublasDaxpy_v2_ptr = get_sym(b"cublasDaxpy_v2\0");
                let cublasSscal_v2_ptr = get_sym(b"cublasSscal_v2\0");
                let cublasDscal_v2_ptr = get_sym(b"cublasDscal_v2\0");
                let cublasSdot_v2_ptr = get_sym(b"cublasSdot_v2\0");
                let cublasDdot_v2_ptr = get_sym(b"cublasDdot_v2\0");
                let cublasSnrm2_v2_ptr = get_sym(b"cublasSnrm2_v2\0");
                let cublasDnrm2_v2_ptr = get_sym(b"cublasDnrm2_v2\0");

                if cublasCreate_v2_ptr.is_null() || cublasSgemm_v2_ptr.is_null() {
                    eprintln!("eigen-rs [cuda]: Required cuBLAS symbols not found.");
                    return None;
                }

                Some(Arc::new(CublasApi {
                    _lib: lib,
                    cublasCreate_v2: std::mem::transmute(cublasCreate_v2_ptr),
                    cublasDestroy_v2: std::mem::transmute(cublasDestroy_v2_ptr),
                    cublasSgemm_v2: std::mem::transmute(cublasSgemm_v2_ptr),
                    cublasDgemm_v2: std::mem::transmute(cublasDgemm_v2_ptr),
                    cublasSaxpy_v2: std::mem::transmute(cublasSaxpy_v2_ptr),
                    cublasDaxpy_v2: std::mem::transmute(cublasDaxpy_v2_ptr),
                    cublasSscal_v2: std::mem::transmute(cublasSscal_v2_ptr),
                    cublasDscal_v2: std::mem::transmute(cublasDscal_v2_ptr),
                    cublasSdot_v2: std::mem::transmute(cublasSdot_v2_ptr),
                    cublasDdot_v2: std::mem::transmute(cublasDdot_v2_ptr),
                    cublasSnrm2_v2: std::mem::transmute(cublasSnrm2_v2_ptr),
                    cublasDnrm2_v2: std::mem::transmute(cublasDnrm2_v2_ptr),
                }))
            }
        })
        .clone()
}

#[cfg(feature = "cuda")]
thread_local! {
    pub static CUBLAS_HANDLE: Option<cublasHandle_t> = {
        if let Some(api) = get_cublas() {
            let mut handle: cublasHandle_t = std::ptr::null_mut();
            unsafe {
                if (api.cublasCreate_v2)(&mut handle) == 0 {
                    Some(handle)
                } else {
                    None
                }
            }
        } else {
            None
        }
    };
}
