#[cfg(feature = "cuda")]
use cuda_sys::cuda::*;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;
use crate::core::storage::CudaStorage;

/// Trait for expressions that can be accelerated on CUDA.
pub trait CudaDispatcher<T: Scalar> {
    /// Returns the underlying CUDA storage if this is a terminal matrix expression.
    fn as_cuda_storage(&self) -> Option<&CudaStorage<T>> {
        None
    }

    fn try_assign_cuda<S: Storage<T>>(&self, _dest: &mut crate::core::matrix::Matrix<T, S>) -> Result<bool, String> {
        Ok(false)
    }
}

#[cfg(feature = "cuda")]
pub struct CudaContext {
    device: CUdevice,
    context: CUcontext,
    module: CUmodule,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaContext {}

#[cfg(feature = "cuda")]
static CUDA_CONTEXT: std::sync::OnceLock<Result<CudaContext, String>> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
pub fn get_cuda_context() -> Result<&'static CudaContext, String> {
    CUDA_CONTEXT.get_or_init(CudaContext::init).as_ref().map_err(|e| e.clone())
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn init() -> Result<Self, String> {
        unsafe {
            let mut device: CUdevice = 0;
            let mut context: CUcontext = std::ptr::null_mut();
            
            if cuInit(0) != CUresult::CUDA_SUCCESS {
                return Err("Failed to initialize CUDA".to_string());
            }
            if cuDeviceGet(&mut device, 0) != CUresult::CUDA_SUCCESS {
                return Err("Failed to get CUDA device".to_string());
            }
            if cuCtxCreate_v2(&mut context, 0, device) != CUresult::CUDA_SUCCESS {
                return Err("Failed to create CUDA context".to_string());
            }

            // Load module from PTX
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
            if ptx.contains("Empty PTX") {
                return Err("CUDA kernels not compiled".to_string());
            }

            let ptx_c_str = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
            let mut module: CUmodule = std::ptr::null_mut();
            
            if cuModuleLoadData(&mut module, ptx_c_str.as_ptr() as *const _) != CUresult::CUDA_SUCCESS {
                return Err("Failed to load CUDA module".to_string());
            }

            Ok(Self { device, context, module })
        }
    }

    pub fn get_function(&self, name: &str) -> Result<CUfunction, String> {
        let name_c_str = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            if cuModuleGetFunction(&mut func, self.module, name_c_str.as_ptr()) != CUresult::CUDA_SUCCESS {
                return Err(format!("Failed to get function: {}", name));
            }
        }
        Ok(func)
    }

    pub unsafe fn launch_add_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        n: i32,
    ) -> Result<(), String> {
        let func = self.get_function("add_kernel_f32")?;
        
        let mut n_val = n;
        let mut args: [*mut std::ffi::c_void; 4] = [
            &a as *const _ as *mut _,
            &b as *const _ as *mut _,
            &c as *const _ as *mut _,
            &mut n_val as *mut _ as *mut _,
        ];

        let threads_per_block = 256;
        let blocks_per_grid = (n as u32 + threads_per_block - 1) / threads_per_block;

        unsafe {
            self.launch_kernel(
                func,
                (blocks_per_grid, 1, 1),
                (threads_per_block, 1, 1),
                &mut args,
            )
        }
    }

    pub unsafe fn launch_sub_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        n: i32,
    ) -> Result<(), String> {
        let func = self.get_function("sub_kernel_f32")?;
        let mut n_val = n;
        let mut args: [*mut std::ffi::c_void; 4] = [
            &a as *const _ as *mut _,
            &b as *const _ as *mut _,
            &c as *const _ as *mut _,
            &mut n_val as *mut _ as *mut _,
        ];
        let threads_per_block = 256;
        let blocks_per_grid = (n as u32 + threads_per_block - 1) / threads_per_block;
        unsafe {
            self.launch_kernel(func, (blocks_per_grid, 1, 1), (threads_per_block, 1, 1), &mut args)
        }
    }

    pub unsafe fn launch_scalar_mul_f32(
        &self,
        a: *const f32,
        s: f32,
        c: *mut f32,
        n: i32,
    ) -> Result<(), String> {
        let func = self.get_function("scalar_mul_kernel_f32")?;
        let mut n_val = n;
        let mut s_val = s;
        let mut args: [*mut std::ffi::c_void; 4] = [
            &a as *const _ as *mut _,
            &mut s_val as *mut _ as *mut _,
            &c as *const _ as *mut _,
            &mut n_val as *mut _ as *mut _,
        ];
        let threads_per_block = 256;
        let blocks_per_grid = (n as u32 + threads_per_block - 1) / threads_per_block;
        unsafe {
            self.launch_kernel(func, (blocks_per_grid, 1, 1), (threads_per_block, 1, 1), &mut args)
        }
    }

    /// Launch a kernel with the given parameters.
    pub unsafe fn launch_kernel(
        &self,
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        unsafe {
            if cuLaunchKernel(
                func,
                grid_dim.0, grid_dim.1, grid_dim.2,
                block_dim.0, block_dim.1, block_dim.2,
                0, std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut()
            ) != CUresult::CUDA_SUCCESS {
                return Err("Failed to launch CUDA kernel".to_string());
            }
            
            if cuCtxSynchronize() != CUresult::CUDA_SUCCESS {
                return Err("CUDA synchronization failed".to_string());
            }
        }
        
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cuCtxDestroy_v2(self.context);
        }
    }
}
