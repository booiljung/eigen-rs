#![allow(non_snake_case)]

#[cfg(feature = "cuda")]
use libloading::Library;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

use super::cublas::cublasFillMode_t;
use super::cublas::cublasOperation_t;

#[allow(non_camel_case_types)]
pub type cusolverStatus_t = i32;
#[allow(non_camel_case_types)]
pub enum cusolverDnHandle {}
#[allow(non_camel_case_types)]
pub type cusolverDnHandle_t = *mut cusolverDnHandle;

pub const CUSOLVER_STATUS_SUCCESS: cusolverStatus_t = 0;

#[cfg(feature = "cuda")]
pub struct CusolverApi {
    _lib: Library,
    pub cusolverDnCreate: unsafe extern "C" fn(*mut cusolverDnHandle_t) -> cusolverStatus_t,
    pub cusolverDnDestroy: unsafe extern "C" fn(cusolverDnHandle_t) -> cusolverStatus_t,
    pub cusolverDnSgetrf_bufferSize: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f32,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDgetrf_bufferSize: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f64,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSgetrf: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f32,
        i32,
        *mut f32,
        *mut i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDgetrf: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f64,
        i32,
        *mut f64,
        *mut i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSgetrs: unsafe extern "C" fn(
        cusolverDnHandle_t,
        cublasOperation_t,
        i32,
        i32,
        *const f32,
        i32,
        *const i32,
        *mut f32,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDgetrs: unsafe extern "C" fn(
        cusolverDnHandle_t,
        cublasOperation_t,
        i32,
        i32,
        *const f64,
        i32,
        *const i32,
        *mut f64,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSpotrf_bufferSize: unsafe extern "C" fn(
        cusolverDnHandle_t,
        cublasFillMode_t,
        i32,
        *mut f32,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDpotrf_bufferSize: unsafe extern "C" fn(
        cusolverDnHandle_t,
        cublasFillMode_t,
        i32,
        *mut f64,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSpotrf: unsafe extern "C" fn(
        cusolverDnHandle_t,
        cublasFillMode_t,
        i32,
        *mut f32,
        i32,
        *mut f32,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDpotrf: unsafe extern "C" fn(
        cusolverDnHandle_t,
        cublasFillMode_t,
        i32,
        *mut f64,
        i32,
        *mut f64,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSgesvd_bufferSize:
        unsafe extern "C" fn(cusolverDnHandle_t, i32, i32, *mut i32) -> cusolverStatus_t,
    pub cusolverDnDgesvd_bufferSize:
        unsafe extern "C" fn(cusolverDnHandle_t, i32, i32, *mut i32) -> cusolverStatus_t,
    pub cusolverDnSgesvd: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i8,
        i8,
        i32,
        i32,
        *mut f32,
        i32,
        *mut f32,
        *mut f32,
        i32,
        *mut f32,
        i32,
        *mut f32,
        i32,
        *mut f32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDgesvd: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i8,
        i8,
        i32,
        i32,
        *mut f64,
        i32,
        *mut f64,
        *mut f64,
        i32,
        *mut f64,
        i32,
        *mut f64,
        i32,
        *mut f64,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSgeqrf_bufferSize: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f32,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDgeqrf_bufferSize: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f64,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnSgeqrf: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f32,
        i32,
        *mut f32,
        *mut f32,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
    pub cusolverDnDgeqrf: unsafe extern "C" fn(
        cusolverDnHandle_t,
        i32,
        i32,
        *mut f64,
        i32,
        *mut f64,
        *mut f64,
        i32,
        *mut i32,
    ) -> cusolverStatus_t,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CusolverApi {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CusolverApi {}

#[cfg(feature = "cuda")]
static CUSOLVER_API: OnceLock<Option<Arc<CusolverApi>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn get_cusolver() -> Option<Arc<CusolverApi>> {
    CUSOLVER_API
        .get_or_init(|| {
            unsafe {
                // macOS / Windows variants could be supported here, but for now linux .so is targeted.
                let lib = match Library::new("libcusolver.so") {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("eigen-rs [cuda]: Failed to load libcusolver.so: {}", e);
                        return None;
                    }
                };

                let get_sym = |name: &[u8]| -> *const () {
                    match lib.get::<*const ()>(name) {
                        Ok(sym) => *sym,
                        Err(_) => std::ptr::null(),
                    }
                };

                let cusolverDnCreate_ptr = get_sym(b"cusolverDnCreate\0");
                let cusolverDnDestroy_ptr = get_sym(b"cusolverDnDestroy\0");
                let cusolverDnSgetrf_bufferSize_ptr = get_sym(b"cusolverDnSgetrf_bufferSize\0");
                let cusolverDnDgetrf_bufferSize_ptr = get_sym(b"cusolverDnDgetrf_bufferSize\0");
                let cusolverDnSgetrf_ptr = get_sym(b"cusolverDnSgetrf\0");
                let cusolverDnDgetrf_ptr = get_sym(b"cusolverDnDgetrf\0");
                let cusolverDnSgetrs_ptr = get_sym(b"cusolverDnSgetrs\0");
                let cusolverDnDgetrs_ptr = get_sym(b"cusolverDnDgetrs\0");

                let cusolverDnSpotrf_bufferSize_ptr = get_sym(b"cusolverDnSpotrf_bufferSize\0");
                let cusolverDnDpotrf_bufferSize_ptr = get_sym(b"cusolverDnDpotrf_bufferSize\0");
                let cusolverDnSpotrf_ptr = get_sym(b"cusolverDnSpotrf\0");
                let cusolverDnDpotrf_ptr = get_sym(b"cusolverDnDpotrf\0");

                let cusolverDnSgesvd_bufferSize_ptr = get_sym(b"cusolverDnSgesvd_bufferSize\0");
                let cusolverDnDgesvd_bufferSize_ptr = get_sym(b"cusolverDnDgesvd_bufferSize\0");
                let cusolverDnSgesvd_ptr = get_sym(b"cusolverDnSgesvd\0");
                let cusolverDnDgesvd_ptr = get_sym(b"cusolverDnDgesvd\0");

                let cusolverDnSgeqrf_bufferSize_ptr = get_sym(b"cusolverDnSgeqrf_bufferSize\0");
                let cusolverDnDgeqrf_bufferSize_ptr = get_sym(b"cusolverDnDgeqrf_bufferSize\0");
                let cusolverDnSgeqrf_ptr = get_sym(b"cusolverDnSgeqrf\0");
                let cusolverDnDgeqrf_ptr = get_sym(b"cusolverDnDgeqrf\0");

                if cusolverDnCreate_ptr.is_null() || cusolverDnSgetrf_ptr.is_null() {
                    eprintln!("eigen-rs [cuda]: Required cuSOLVER symbols not found.");
                    return None;
                }

                Some(Arc::new(CusolverApi {
                    _lib: lib,
                    cusolverDnCreate: std::mem::transmute(cusolverDnCreate_ptr),
                    cusolverDnDestroy: std::mem::transmute(cusolverDnDestroy_ptr),
                    cusolverDnSgetrf_bufferSize: std::mem::transmute(
                        cusolverDnSgetrf_bufferSize_ptr,
                    ),
                    cusolverDnDgetrf_bufferSize: std::mem::transmute(
                        cusolverDnDgetrf_bufferSize_ptr,
                    ),
                    cusolverDnSgetrf: std::mem::transmute(cusolverDnSgetrf_ptr),
                    cusolverDnDgetrf: std::mem::transmute(cusolverDnDgetrf_ptr),
                    cusolverDnSgetrs: std::mem::transmute(cusolverDnSgetrs_ptr),
                    cusolverDnDgetrs: std::mem::transmute(cusolverDnDgetrs_ptr),
                    cusolverDnSpotrf_bufferSize: std::mem::transmute(
                        cusolverDnSpotrf_bufferSize_ptr,
                    ),
                    cusolverDnDpotrf_bufferSize: std::mem::transmute(
                        cusolverDnDpotrf_bufferSize_ptr,
                    ),
                    cusolverDnSpotrf: std::mem::transmute(cusolverDnSpotrf_ptr),
                    cusolverDnDpotrf: std::mem::transmute(cusolverDnDpotrf_ptr),
                    cusolverDnSgesvd_bufferSize: std::mem::transmute(
                        cusolverDnSgesvd_bufferSize_ptr,
                    ),
                    cusolverDnDgesvd_bufferSize: std::mem::transmute(
                        cusolverDnDgesvd_bufferSize_ptr,
                    ),
                    cusolverDnSgesvd: std::mem::transmute(cusolverDnSgesvd_ptr),
                    cusolverDnDgesvd: std::mem::transmute(cusolverDnDgesvd_ptr),
                    cusolverDnSgeqrf_bufferSize: std::mem::transmute(
                        cusolverDnSgeqrf_bufferSize_ptr,
                    ),
                    cusolverDnDgeqrf_bufferSize: std::mem::transmute(
                        cusolverDnDgeqrf_bufferSize_ptr,
                    ),
                    cusolverDnSgeqrf: std::mem::transmute(cusolverDnSgeqrf_ptr),
                    cusolverDnDgeqrf: std::mem::transmute(cusolverDnDgeqrf_ptr),
                }))
            }
        })
        .clone()
}

#[cfg(feature = "cuda")]
thread_local! {
    pub static CUSOLVER_HANDLE: Option<cusolverDnHandle_t> = {
        if let Some(api) = get_cusolver() {
            let mut handle: cusolverDnHandle_t = std::ptr::null_mut();
            unsafe {
                if (api.cusolverDnCreate)(&mut handle) == 0 {
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
