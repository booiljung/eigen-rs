//! cuSPARSE wrappers for sparse matrix operations on GPU.

use crate::core::scalar::Scalar;
use crate::core::sparse::cuda_storage::CudaSparseStorage;
use crate::core::storage::CudaStorage;

#[cfg(feature = "cuda")]
pub mod sys {
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cusparseContext {
        _unused: [u8; 0],
    }
    pub type cusparseHandle_t = *mut cusparseContext;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cusparseSpMatDescr {
        _unused: [u8; 0],
    }
    pub type cusparseSpMatDescr_t = *mut cusparseSpMatDescr;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cusparseDnVecDescr {
        _unused: [u8; 0],
    }
    pub type cusparseDnVecDescr_t = *mut cusparseDnVecDescr;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cusparseDnMatDescr {
        _unused: [u8; 0],
    }
    pub type cusparseDnMatDescr_t = *mut cusparseDnMatDescr;

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum cusparseStatus_t {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        InvalidValue = 3,
        ArchMismatch = 4,
        MappingError = 5,
        ExecutionFailed = 6,
        InternalError = 7,
        MatrixTypeNotSupported = 8,
        ZeroPivot = 9,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum cusparseIndexType_t {
        I16 = 1,
        I32 = 2,
        I64 = 3,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum cusparseSpMVAlg_t {
        Default = 0,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum cusparseSpMMAlg_t {
        Default = 0,
    }

    unsafe extern "C" {
        pub fn cusparseCreate(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
        pub fn cusparseDestroy(handle: cusparseHandle_t) -> cusparseStatus_t;

        pub fn cusparseCreateCsr(
            spMatDescr: *mut cusparseSpMatDescr_t,
            rows: i64,
            cols: i64,
            nnz: i64,
            rowOffsets: *mut std::ffi::c_void,
            colIndices: *mut std::ffi::c_void,
            values: *mut std::ffi::c_void,
            rowOffsetsType: cusparseIndexType_t,
            colIndicesType: cusparseIndexType_t,
            indexBase: i32, // 0 or 1
            valueType: i32, // CUDA_R_32F = 0, CUDA_R_64F = 1
        ) -> cusparseStatus_t;

        pub fn cusparseDestroySpMat(spMatDescr: cusparseSpMatDescr_t) -> cusparseStatus_t;

        pub fn cusparseCreateDnVec(
            dnVecDescr: *mut cusparseDnVecDescr_t,
            size: i64,
            values: *mut std::ffi::c_void,
            valueType: i32,
        ) -> cusparseStatus_t;

        pub fn cusparseDestroyDnVec(dnVecDescr: cusparseDnVecDescr_t) -> cusparseStatus_t;

        pub fn cusparseCreateDnMat(
            dnMatDescr: *mut cusparseDnMatDescr_t,
            rows: i64,
            cols: i64,
            ld: i64,
            values: *mut std::ffi::c_void,
            valueType: i32,
            order: i32, // CUSPARSE_ORDER_COL = 1
        ) -> cusparseStatus_t;

        pub fn cusparseDestroyDnMat(dnMatDescr: cusparseDnMatDescr_t) -> cusparseStatus_t;

        pub fn cusparseSpMV(
            handle: cusparseHandle_t,
            opA: i32,
            alpha: *const std::ffi::c_void,
            spMatA: cusparseSpMatDescr_t,
            dnVecX: cusparseDnVecDescr_t,
            beta: *const std::ffi::c_void,
            dnVecY: cusparseDnVecDescr_t,
            computeType: i32,
            alg: cusparseSpMVAlg_t,
            externalBuffer: *mut std::ffi::c_void,
        ) -> cusparseStatus_t;

        pub fn cusparseSpMV_bufferSize(
            handle: cusparseHandle_t,
            opA: i32,
            alpha: *const std::ffi::c_void,
            spMatA: cusparseSpMatDescr_t,
            dnVecX: cusparseDnVecDescr_t,
            beta: *const std::ffi::c_void,
            dnVecY: cusparseDnVecDescr_t,
            computeType: i32,
            alg: cusparseSpMVAlg_t,
            bufferSize: *mut usize,
        ) -> cusparseStatus_t;

        pub fn cusparseSpMM(
            handle: cusparseHandle_t,
            opA: i32,
            opB: i32,
            alpha: *const std::ffi::c_void,
            spMatA: cusparseSpMatDescr_t,
            dnMatB: cusparseDnMatDescr_t,
            beta: *const std::ffi::c_void,
            dnMatC: cusparseDnMatDescr_t,
            computeType: i32,
            alg: cusparseSpMMAlg_t,
            externalBuffer: *mut std::ffi::c_void,
        ) -> cusparseStatus_t;

        pub fn cusparseSpMM_bufferSize(
            handle: cusparseHandle_t,
            opA: i32,
            opB: i32,
            alpha: *const std::ffi::c_void,
            spMatA: cusparseSpMatDescr_t,
            dnMatB: cusparseDnMatDescr_t,
            beta: *const std::ffi::c_void,
            dnMatC: cusparseDnMatDescr_t,
            computeType: i32,
            alg: cusparseSpMMAlg_t,
            bufferSize: *mut usize,
        ) -> cusparseStatus_t;
    }
}

#[cfg(not(feature = "cuda"))]
pub mod sys {
    #[allow(non_camel_case_types)]
    pub type cusparseHandle_t = ();
    #[allow(non_camel_case_types)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum cusparseStatus_t {
        Success = 0,
    }
}

pub struct CusparseHandle {
    #[cfg(feature = "cuda")]
    handle: sys::cusparseHandle_t,
    #[cfg(not(feature = "cuda"))]
    _handle: sys::cusparseHandle_t,
}

impl CusparseHandle {
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            let mut handle = std::ptr::null_mut();
            unsafe {
                let res = sys::cusparseCreate(&mut handle);
                if res != sys::cusparseStatus_t::Success {
                    return Err(format!("cusparseCreate failed: {:?}", res));
                }
            }
            Ok(Self { handle })
        }
        #[cfg(not(feature = "cuda"))]
        Err("CUDA not enabled".to_string())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CusparseHandle {
    fn drop(&mut self) {
        unsafe {
            sys::cusparseDestroy(self.handle);
        }
    }
}

/// Perform Y = Alpha * A * X + Beta * Y on GPU
pub fn spmv_cuda<T: Scalar>(
    handle: &CusparseHandle,
    a: &CudaSparseStorage<T>,
    x: &CudaStorage<T>,
    y: &mut CudaStorage<T>,
    alpha: T,
    beta: T,
) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    unsafe {
        let mut mat_a = std::ptr::null_mut();
        let value_type = if std::mem::size_of::<T>() == 8 { 1 } else { 0 };

        let res = sys::cusparseCreateCsr(
            &mut mat_a,
            a.rows as i64,
            a.cols as i64,
            a.nnz as i64,
            a.row_offsets.as_ptr() as *mut _,
            a.col_indices.as_ptr() as *mut _,
            a.values.as_ptr() as *mut _,
            sys::cusparseIndexType_t::I32,
            sys::cusparseIndexType_t::I32,
            0,
            value_type,
        );
        if res != sys::cusparseStatus_t::Success {
            return Err("cusparseCreateCsr failed".to_string());
        }

        let mut vec_x = std::ptr::null_mut();
        sys::cusparseCreateDnVec(
            &mut vec_x,
            x.rows() as i64,
            x.get_ptr(0, 0) as *mut _,
            value_type,
        );

        let mut vec_y = std::ptr::null_mut();
        sys::cusparseCreateDnVec(
            &mut vec_y,
            y.rows() as i64,
            y.get_ptr(0, 0) as *mut _,
            value_type,
        );

        let mut buffer_size = 0;
        sys::cusparseSpMV_bufferSize(
            handle.handle,
            0,
            &alpha as *const T as *const _,
            mat_a,
            vec_x,
            &beta as *const T as *const _,
            vec_y,
            value_type,
            sys::cusparseSpMVAlg_t::Default,
            &mut buffer_size,
        );

        let mut buffer = std::ptr::null_mut();
        if buffer_size > 0 {
            use cuda_sys::cudart::cudaMalloc;
            cudaMalloc(&mut buffer, buffer_size);
        }

        let res = sys::cusparseSpMV(
            handle.handle,
            0,
            &alpha as *const T as *const _,
            mat_a,
            vec_x,
            &beta as *const T as *const _,
            vec_y,
            value_type,
            sys::cusparseSpMVAlg_t::Default,
            buffer,
        );

        if buffer_size > 0 {
            use cuda_sys::cudart::cudaFree;
            cudaFree(buffer);
        }

        sys::cusparseDestroySpMat(mat_a);
        sys::cusparseDestroyDnVec(vec_x);
        sys::cusparseDestroyDnVec(vec_y);

        if res == sys::cusparseStatus_t::Success {
            Ok(())
        } else {
            Err(format!("cusparseSpMV failed: {:?}", res))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (handle, a, x, y, alpha, beta);
        Err("CUDA not enabled".to_string())
    }
}

/// Perform C = Alpha * A * B + Beta * C on GPU (SpMM)
pub fn spmm_cuda<T: Scalar>(
    handle: &CusparseHandle,
    a: &CudaSparseStorage<T>,
    b: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
    alpha: T,
    beta: T,
) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    unsafe {
        let mut mat_a = std::ptr::null_mut();
        let value_type = if std::mem::size_of::<T>() == 8 { 1 } else { 0 };

        sys::cusparseCreateCsr(
            &mut mat_a,
            a.rows as i64,
            a.cols as i64,
            a.nnz as i64,
            a.row_offsets.as_ptr() as *mut _,
            a.col_indices.as_ptr() as *mut _,
            a.values.as_ptr() as *mut _,
            sys::cusparseIndexType_t::I32,
            sys::cusparseIndexType_t::I32,
            0,
            value_type,
        );

        let mut mat_b = std::ptr::null_mut();
        sys::cusparseCreateDnMat(
            &mut mat_b,
            b.rows() as i64,
            b.cols() as i64,
            b.rows() as i64,
            b.get_ptr(0, 0) as *mut _,
            value_type,
            1,
        );

        let mut mat_c = std::ptr::null_mut();
        sys::cusparseCreateDnMat(
            &mut mat_c,
            c.rows() as i64,
            c.cols() as i64,
            c.rows() as i64,
            c.get_ptr(0, 0) as *mut _,
            value_type,
            1,
        );

        let mut buffer_size = 0;
        sys::cusparseSpMM_bufferSize(
            handle.handle,
            0,
            0,
            &alpha as *const T as *const _,
            mat_a,
            mat_b,
            &beta as *const T as *const _,
            mat_c,
            value_type,
            sys::cusparseSpMMAlg_t::Default,
            &mut buffer_size,
        );

        let mut buffer = std::ptr::null_mut();
        if buffer_size > 0 {
            use cuda_sys::cudart::cudaMalloc;
            cudaMalloc(&mut buffer, buffer_size);
        }

        let res = sys::cusparseSpMM(
            handle.handle,
            0,
            0,
            &alpha as *const T as *const _,
            mat_a,
            mat_b,
            &beta as *const T as *const _,
            mat_c,
            value_type,
            sys::cusparseSpMMAlg_t::Default,
            buffer,
        );

        if buffer_size > 0 {
            use cuda_sys::cudart::cudaFree;
            cudaFree(buffer);
        }

        sys::cusparseDestroySpMat(mat_a);
        sys::cusparseDestroyDnMat(mat_b);
        sys::cusparseDestroyDnMat(mat_c);

        if res == sys::cusparseStatus_t::Success {
            Ok(())
        } else {
            Err(format!("cusparseSpMM failed: {:?}", res))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (handle, a, b, c, alpha, beta);
        Err("CUDA not enabled".to_string())
    }
}
