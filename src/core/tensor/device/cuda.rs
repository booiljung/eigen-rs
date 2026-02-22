//! CUDA Device implementation using `cudart` dynamic linkage.

use crate::core::scalar::Scalar;
use crate::core::tensor::device::{Device, DeviceStorage};

#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// CUDA Device implementation.
///
/// Operates as a logical target for tensors that want to reside in GPU memory.
#[derive(Clone, Debug)]
pub struct CudaDevice {
    #[allow(dead_code)]
    device_id: u32,
}

#[cfg(feature = "cuda")]
pub fn is_cuda_device_active() -> bool {
    if let Some(api) = crate::core::tensor::device::cudart::get_cudart() {
        let mut count = 0;
        if unsafe { (api.cudaGetDeviceCount)(&mut count) } == 0 {
            return count > 0;
        }
    }
    false
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self { device_id: 0 }
    }
}

impl Device for CudaDevice {
    type Storage<T: Scalar> = CudaStorage<T>;

    fn name(&self) -> &'static str {
        "CUDA"
    }
}

/// Storage in CUDA Device Memory.
pub struct CudaStorage<T: Scalar> {
    #[cfg(feature = "cuda")]
    ptr: *mut c_void,
    _marker: std::marker::PhantomData<T>,
    size: usize,
}

#[cfg(feature = "cuda")]
impl<T: Scalar> Drop for CudaStorage<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = crate::core::tensor::device::cudart::cuda_free(self.ptr);
            self.ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(feature = "cuda")]
impl<T: Scalar> DeviceStorage<T> for CudaStorage<T> {
    fn new(size: usize) -> Result<Self, String> {
        let bytes = size * std::mem::size_of::<T>();
        let ptr = crate::core::tensor::device::cudart::cuda_malloc(bytes)?;

        Ok(Self {
            ptr,
            _marker: std::marker::PhantomData,
            size,
        })
    }

    fn as_slice(&self) -> Option<&[T]> {
        None
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn copy_from_host(&mut self, src: &[T]) -> Result<(), String> {
        if src.len() != self.size {
            return Err("Size mismatch".into());
        }
        let bytes = self.size * std::mem::size_of::<T>();
        crate::core::tensor::device::cudart::cuda_memcpy_h2d(
            self.ptr,
            src.as_ptr() as *const _,
            bytes,
        )
    }

    fn copy_to_host(&self, dest: &mut [T]) -> Result<(), String> {
        if dest.len() != self.size {
            return Err("Size mismatch".into());
        }
        let bytes = self.size * std::mem::size_of::<T>();
        crate::core::tensor::device::cudart::cuda_memcpy_d2h(
            dest.as_mut_ptr() as *mut _,
            self.ptr,
            bytes,
        )
    }
}

#[cfg(not(feature = "cuda"))]
impl<T: Scalar> DeviceStorage<T> for CudaStorage<T> {
    fn new(_size: usize) -> Result<Self, String> {
        Err("CUDA feature not enabled".to_string())
    }
    fn as_slice(&self) -> Option<&[T]> {
        None
    }
    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }
    fn copy_from_host(&mut self, _src: &[T]) -> Result<(), String> {
        Err("CUDA feature not enabled".to_string())
    }
    fn copy_to_host(&self, _dest: &mut [T]) -> Result<(), String> {
        Err("CUDA feature not enabled".to_string())
    }
}

#[cfg(feature = "cuda")]
impl<T: Scalar> CudaStorage<T> {
    pub fn copy_device_to_device(&mut self, src: &CudaStorage<T>) -> Result<(), String> {
        if self.size != src.size {
            return Err("Size mismatch".into());
        }
        Err("CUDA D2D copy not yet supported via dynamic bridge".into())
    }

    pub fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

// Manual implementation of AsMut/AsRef which will panic, ensuring safety at runtime
// if user tries to treat it as CPU memory.
// Ideally, we refactor DeviceStorage to not require AsMut<[T]> but for now checking runtime panic.
impl<T: Scalar> AsRef<[T]> for CudaStorage<T> {
    fn as_ref(&self) -> &[T] {
        panic!("CudaStorage cannot be used as &[T]");
    }
}

impl<T: Scalar> AsMut<[T]> for CudaStorage<T> {
    fn as_mut(&mut self) -> &mut [T] {
        panic!("CudaStorage cannot be used as &mut [T]");
    }
}

impl CudaDevice {
    #[cfg(feature = "cuda")]
    pub fn assign<T: Scalar>(
        &self,
        _out: &mut CudaStorage<T>,
        _inp: &CudaStorage<T>,
    ) -> Result<(), String> {
        Err("Elemental assign on CUDA not supported without PTX module".into())
    }

    #[cfg(feature = "cuda")]
    pub fn add<T: Scalar>(
        &self,
        out: &mut CudaStorage<T>,
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
    ) -> Result<(), String> {
        let size = a.size;
        if size != out.size || size != b.size {
            return Err("CUDA add size mismatch".to_string());
        }
        crate::core::tensor::device::cudart::cuda_memcpy_d2d(
            out.as_device_ptr(),
            b.as_device_ptr() as *const _,
            size * std::mem::size_of::<T>(),
        )?;

        let handle_opt = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuBLAS handle not initialized")?;
        let api_opt = crate::core::tensor::device::cublas::get_cublas();
        let api = api_opt.ok_or("cuBLAS library not loaded")?;

        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            let alpha = 1.0f32;
            let status = unsafe {
                (api.cublasSaxpy_v2)(
                    handle,
                    size as i32,
                    &alpha as *const f32,
                    a.as_device_ptr() as *const f32,
                    1,
                    out.as_device_ptr() as *mut f32,
                    1,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSaxpy failed: {}", status));
            }
        } else {
            let alpha = 1.0f64;
            let status = unsafe {
                (api.cublasDaxpy_v2)(
                    handle,
                    size as i32,
                    &alpha as *const f64,
                    a.as_device_ptr() as *const f64,
                    1,
                    out.as_device_ptr() as *mut f64,
                    1,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasDaxpy failed: {}", status));
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn sub<T: Scalar>(
        &self,
        out: &mut CudaStorage<T>,
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
    ) -> Result<(), String> {
        let size = a.size;
        if size != out.size || size != b.size {
            return Err("CUDA sub size mismatch".to_string());
        }
        crate::core::tensor::device::cudart::cuda_memcpy_d2d(
            out.as_device_ptr(),
            a.as_device_ptr() as *const _,
            size * std::mem::size_of::<T>(),
        )?;

        let handle_opt = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuBLAS handle not initialized")?;
        let api_opt = crate::core::tensor::device::cublas::get_cublas();
        let api = api_opt.ok_or("cuBLAS library not loaded")?;

        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            let alpha = -1.0f32;
            let status = unsafe {
                (api.cublasSaxpy_v2)(
                    handle,
                    size as i32,
                    &alpha as *const f32,
                    b.as_device_ptr() as *const f32,
                    1,
                    out.as_device_ptr() as *mut f32,
                    1,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSaxpy failed: {}", status));
            }
        } else {
            let alpha = -1.0f64;
            let status = unsafe {
                (api.cublasDaxpy_v2)(
                    handle,
                    size as i32,
                    &alpha as *const f64,
                    b.as_device_ptr() as *const f64,
                    1,
                    out.as_device_ptr() as *mut f64,
                    1,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasDaxpy failed: {}", status));
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn mul_scalar<T: Scalar>(
        &self,
        out: &mut CudaStorage<T>,
        inp: &CudaStorage<T>,
        scalar: T,
    ) -> Result<(), String> {
        let size = inp.size;
        if size != out.size {
            return Err("CUDA mul_scalar size mismatch".to_string());
        }
        crate::core::tensor::device::cudart::cuda_memcpy_d2d(
            out.as_device_ptr(),
            inp.as_device_ptr() as *const _,
            size * std::mem::size_of::<T>(),
        )?;

        let handle_opt = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuBLAS handle not initialized")?;
        let api_opt = crate::core::tensor::device::cublas::get_cublas();
        let api = api_opt.ok_or("cuBLAS library not loaded")?;

        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            let val: f32 = unsafe { std::mem::transmute_copy(&scalar) };
            let status = unsafe {
                (api.cublasSscal_v2)(
                    handle,
                    size as i32,
                    &val as *const f32,
                    out.as_device_ptr() as *mut f32,
                    1,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSscal failed: {}", status));
            }
        } else {
            let val: f64 = unsafe { std::mem::transmute_copy(&scalar) };
            let status = unsafe {
                (api.cublasDscal_v2)(
                    handle,
                    size as i32,
                    &val as *const f64,
                    out.as_device_ptr() as *mut f64,
                    1,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasDscal failed: {}", status));
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn matmul<T: Scalar>(
        &self,
        _c: &mut CudaStorage<T>,
        _a: &CudaStorage<T>,
        _b: &CudaStorage<T>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), String> {
        // We offload matmul to cuBLAS directly inside gemm_blocked instead of elemental kernels
        Err("matmul on CudaDevice natively via kernel is unsupported. Use global cuBLAS hook in gemm_blocked.".into())
    }

    #[cfg(feature = "cuda")]
    pub fn permute<T: Scalar>(
        &self,
        _out: &mut CudaStorage<T>,
        _inp: &CudaStorage<T>,
        _rank: usize,
        _out_dims: &[usize],
        _out_strides: &[usize],
        _in_strides: &[usize],
        _perm: &[usize],
    ) -> Result<(), String> {
        Err("Elemental permute on CUDA not supported without PTX module".into())
    }

    #[cfg(feature = "cuda")]
    pub fn dot<T: Scalar>(
        &self,
        a: &CudaStorage<T>,
        b: &CudaStorage<T>,
    ) -> Result<T, String> {
        let size = a.size;
        if size != b.size {
            return Err("CUDA dot size mismatch".to_string());
        }

        let handle_opt = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuBLAS handle not initialized")?;
        let api_opt = crate::core::tensor::device::cublas::get_cublas();
        let api = api_opt.ok_or("cuBLAS library not loaded")?;

        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            let mut result = 0.0f32;
            let status = unsafe {
                (api.cublasSdot_v2)(
                    handle,
                    size as i32,
                    a.as_device_ptr() as *const f32,
                    1,
                    b.as_device_ptr() as *const f32,
                    1,
                    &mut result as *mut f32,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSdot failed: {}", status));
            }
            Ok(T::from_f64(result as f64))
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let mut result = 0.0f64;
            let status = unsafe {
                (api.cublasDdot_v2)(
                    handle,
                    size as i32,
                    a.as_device_ptr() as *const f64,
                    1,
                    b.as_device_ptr() as *const f64,
                    1,
                    &mut result as *mut f64,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasDdot failed: {}", status));
            }
            Ok(T::from_f64(result))
        } else {
            Err("CUDA dot unsupported type".to_string())
        }
    }

    #[cfg(feature = "cuda")]
    pub fn norm<T: Scalar>(
        &self,
        a: &CudaStorage<T>,
    ) -> Result<T, String> {
        let size = a.size;

        let handle_opt = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuBLAS handle not initialized")?;
        let api_opt = crate::core::tensor::device::cublas::get_cublas();
        let api = api_opt.ok_or("cuBLAS library not loaded")?;

        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            let mut result = 0.0f32;
            let status = unsafe {
                (api.cublasSnrm2_v2)(
                    handle,
                    size as i32,
                    a.as_device_ptr() as *const f32,
                    1,
                    &mut result as *mut f32,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSnrm2 failed: {}", status));
            }
            Ok(T::from_f64(result as f64))
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let mut result = 0.0f64;
            let status = unsafe {
                (api.cublasDnrm2_v2)(
                    handle,
                    size as i32,
                    a.as_device_ptr() as *const f64,
                    1,
                    &mut result as *mut f64,
                )
            };
            if status != crate::core::tensor::device::cublas::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasDnrm2 failed: {}", status));
            }
            Ok(T::from_f64(result))
        } else {
            Err("CUDA norm unsupported type".to_string())
        }
    }
}
