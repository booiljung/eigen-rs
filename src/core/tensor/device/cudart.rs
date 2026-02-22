#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[cfg(feature = "cuda")]
use libloading::{Library, Symbol};
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
#[allow(non_camel_case_types)]
pub type cudaError_t = i32;

#[cfg(feature = "cuda")]
pub const CUDA_SUCCESS: cudaError_t = 0;

#[cfg(feature = "cuda")]
pub const cudaMemcpyHostToDevice: i32 = 1;
#[cfg(feature = "cuda")]
pub const cudaMemcpyDeviceToHost: i32 = 2;
#[cfg(feature = "cuda")]
pub const cudaMemcpyDeviceToDevice: i32 = 3;

#[cfg(feature = "cuda")]
pub struct CudartApi {
    _lib: Library,
    pub cudaMalloc: unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize) -> cudaError_t,
    pub cudaFree: unsafe extern "C" fn(*mut std::ffi::c_void) -> cudaError_t,
    pub cudaMemcpy: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const std::ffi::c_void,
        usize,
        i32,
    ) -> cudaError_t,
    pub cudaDeviceSynchronize: unsafe extern "C" fn() -> cudaError_t,
    pub cudaGetDeviceCount: unsafe extern "C" fn(*mut i32) -> cudaError_t,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudartApi {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudartApi {}

#[cfg(feature = "cuda")]
static CUDART_API: OnceLock<Option<Arc<CudartApi>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn get_cudart() -> Option<Arc<CudartApi>> {
    CUDART_API
        .get_or_init(|| unsafe {
            let lib = match Library::new("libcudart.so") {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("eigen-rs [cuda]: Failed to load libcudart.so: {}", e);
                    return None;
                }
            };

            let get_sym = |name: &[u8]| -> *const () {
                match lib.get::<*const ()>(name) {
                    Ok(sym) => *sym,
                    Err(_) => std::ptr::null(),
                }
            };

            let cudaMalloc_ptr = get_sym(b"cudaMalloc\0");
            let cudaFree_ptr = get_sym(b"cudaFree\0");
            let cudaMemcpy_ptr = get_sym(b"cudaMemcpy\0");
            let cudaDeviceSynchronize_ptr = get_sym(b"cudaDeviceSynchronize\0");
            let cudaGetDeviceCount_ptr = get_sym(b"cudaGetDeviceCount\0");

            if cudaMalloc_ptr.is_null() || cudaMemcpy_ptr.is_null() {
                eprintln!("eigen-rs [cuda]: Required CUDART symbols not found.");
                return None;
            }

            Some(Arc::new(CudartApi {
                _lib: lib,
                cudaMalloc: std::mem::transmute(cudaMalloc_ptr),
                cudaFree: std::mem::transmute(cudaFree_ptr),
                cudaMemcpy: std::mem::transmute(cudaMemcpy_ptr),
                cudaDeviceSynchronize: std::mem::transmute(cudaDeviceSynchronize_ptr),
                cudaGetDeviceCount: std::mem::transmute(cudaGetDeviceCount_ptr),
            }))
        })
        .clone()
}

#[cfg(feature = "cuda")]
pub fn cuda_malloc(size_bytes: usize) -> Result<*mut std::ffi::c_void, String> {
    if let Some(api) = get_cudart() {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe { (api.cudaMalloc)(&mut ptr, size_bytes) };
        if status == CUDA_SUCCESS {
            Ok(ptr)
        } else {
            Err(format!("cudaMalloc failed: {}", status))
        }
    } else {
        Err("libcudart.so not loaded".to_string())
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_free(ptr: *mut std::ffi::c_void) -> Result<(), String> {
    if ptr.is_null() {
        return Ok(());
    }
    if let Some(api) = get_cudart() {
        let status = unsafe { (api.cudaFree)(ptr) };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("cudaFree failed: {}", status))
        }
    } else {
        Err("libcudart.so not loaded".to_string())
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_memcpy_h2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size_bytes: usize,
) -> Result<(), String> {
    if let Some(api) = get_cudart() {
        let status = unsafe { (api.cudaMemcpy)(dst, src, size_bytes, cudaMemcpyHostToDevice) };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("cudaMemcpy H2D failed: {}", status))
        }
    } else {
        Err("libcudart.so not loaded".to_string())
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_memcpy_d2h(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size_bytes: usize,
) -> Result<(), String> {
    if let Some(api) = get_cudart() {
        let status = unsafe { (api.cudaMemcpy)(dst, src, size_bytes, cudaMemcpyDeviceToHost) };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("cudaMemcpy D2H failed: {}", status))
        }
    } else {
        Err("libcudart.so not loaded".to_string())
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_memcpy_d2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size_bytes: usize,
) -> Result<(), String> {
    if let Some(api) = get_cudart() {
        let status = unsafe { (api.cudaMemcpy)(dst, src, size_bytes, cudaMemcpyDeviceToDevice) };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("cudaMemcpy D2D failed: {}", status))
        }
    } else {
        Err("libcudart.so not loaded".to_string())
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_device_synchronize() -> Result<(), String> {
    if let Some(api) = get_cudart() {
        let status = unsafe { (api.cudaDeviceSynchronize)() };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("cudaDeviceSynchronize failed: {}", status))
        }
    } else {
        Err("libcudart.so not loaded".to_string())
    }
}
