#![allow(non_snake_case)]

#[cfg(feature = "cuda")]
use libloading::Library;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[allow(non_camel_case_types)]
pub type cusparseStatus_t = i32;

pub const CUSPARSE_STATUS_SUCCESS: cusparseStatus_t = 0;

#[allow(non_camel_case_types)]
pub enum cusparseContext {}
#[allow(non_camel_case_types)]
pub type cusparseHandle_t = *mut cusparseContext;

#[allow(non_camel_case_types)]
pub enum cusparseSpMatDescr {}
#[allow(non_camel_case_types)]
pub type cusparseSpMatDescr_t = *mut cusparseSpMatDescr;

#[allow(non_camel_case_types)]
pub enum cusparseDnVecDescr {}
#[allow(non_camel_case_types)]
pub type cusparseDnVecDescr_t = *mut cusparseDnVecDescr;

#[allow(non_camel_case_types)]
pub enum cusparseDnMatDescr {}
#[allow(non_camel_case_types)]
pub type cusparseDnMatDescr_t = *mut cusparseDnMatDescr;

#[allow(non_camel_case_types)]
pub type cusparseIndexType_t = i32;
pub const CUSPARSE_INDEX_32I: cusparseIndexType_t = 1;

#[allow(non_camel_case_types)]
pub type cusparseIndexBase_t = i32;
pub const CUSPARSE_INDEX_BASE_ZERO: cusparseIndexBase_t = 0;

#[allow(non_camel_case_types)]
pub type cusparseOperation_t = i32;
pub const CUSPARSE_OPERATION_NON_TRANSPOSE: cusparseOperation_t = 0;
pub const CUSPARSE_OPERATION_TRANSPOSE: cusparseOperation_t = 1;

#[allow(non_camel_case_types)]
pub type cudaDataType = i32;
pub const CUDA_R_32F: cudaDataType = 0;
pub const CUDA_R_64F: cudaDataType = 1;

#[allow(non_camel_case_types)]
pub type cusparseSpMVAlg_t = i32;
pub const CUSPARSE_SPMV_ALG_DEFAULT: cusparseSpMVAlg_t = 0;

#[allow(non_camel_case_types)]
pub type cusparseSpMMAlg_t = i32;
pub const CUSPARSE_SPMM_ALG_DEFAULT: cusparseSpMMAlg_t = 0;

#[cfg(feature = "cuda")]
pub struct CusparseApi {
    _lib: Library,
    pub cusparseCreate: unsafe extern "C" fn(*mut cusparseHandle_t) -> cusparseStatus_t,
    pub cusparseDestroy: unsafe extern "C" fn(cusparseHandle_t) -> cusparseStatus_t,
    pub cusparseCreateCsr: unsafe extern "C" fn(
        *mut cusparseSpMatDescr_t,
        i64, // rows
        i64, // cols
        i64, // nnz
        *mut std::ffi::c_void, // csrRowOffsets
        *mut std::ffi::c_void, // csrColInd
        *mut std::ffi::c_void, // csrValues
        cusparseIndexType_t, // csrRowOffsetsType
        cusparseIndexType_t, // csrColIndType
        cusparseIndexBase_t, // idxBase
        cudaDataType, // valueType
    ) -> cusparseStatus_t,
    pub cusparseCreateDnVec: unsafe extern "C" fn(
        *mut cusparseDnVecDescr_t,
        i64, // size
        *mut std::ffi::c_void, // values
        cudaDataType, // valueType
    ) -> cusparseStatus_t,
    pub cusparseDestroySpMat: unsafe extern "C" fn(cusparseSpMatDescr_t) -> cusparseStatus_t,
    pub cusparseDestroyDnVec: unsafe extern "C" fn(cusparseDnVecDescr_t) -> cusparseStatus_t,
    pub cusparseCreateDnMat: unsafe extern "C" fn(
        *mut cusparseDnMatDescr_t,
        i64, // rows
        i64, // cols
        i64, // ld
        *mut std::ffi::c_void, // values
        cudaDataType, // valueType
        i32, // order
    ) -> cusparseStatus_t,
    pub cusparseDestroyDnMat: unsafe extern "C" fn(cusparseDnMatDescr_t) -> cusparseStatus_t,
    pub cusparseSpMV_bufferSize: unsafe extern "C" fn(
        cusparseHandle_t,
        cusparseOperation_t, // opA
        *const std::ffi::c_void, // alpha
        cusparseSpMatDescr_t, // matA
        cusparseDnVecDescr_t, // vecX
        *const std::ffi::c_void, // beta
        cusparseDnVecDescr_t, // vecY
        cudaDataType, // computeType
        cusparseSpMVAlg_t, // alg
        *mut usize, // bufferSize
    ) -> cusparseStatus_t,
    pub cusparseSpMV: unsafe extern "C" fn(
        cusparseHandle_t,
        cusparseOperation_t, // opA
        *const std::ffi::c_void, // alpha
        cusparseSpMatDescr_t, // matA
        cusparseDnVecDescr_t, // vecX
        *const std::ffi::c_void, // beta
        cusparseDnVecDescr_t, // vecY
        cudaDataType, // computeType
        cusparseSpMVAlg_t, // alg
        *mut std::ffi::c_void, // externalBuffer
    ) -> cusparseStatus_t,
    pub cusparseSpMM_bufferSize: unsafe extern "C" fn(
        cusparseHandle_t,
        cusparseOperation_t, // opA
        cusparseOperation_t, // opB
        *const std::ffi::c_void, // alpha
        cusparseSpMatDescr_t, // matA
        cusparseDnMatDescr_t, // matB
        *const std::ffi::c_void, // beta
        cusparseDnMatDescr_t, // matC
        cudaDataType, // computeType
        cusparseSpMMAlg_t, // alg
        *mut usize, // bufferSize
    ) -> cusparseStatus_t,
    pub cusparseSpMM: unsafe extern "C" fn(
        cusparseHandle_t,
        cusparseOperation_t, // opA
        cusparseOperation_t, // opB
        *const std::ffi::c_void, // alpha
        cusparseSpMatDescr_t, // matA
        cusparseDnMatDescr_t, // matB
        *const std::ffi::c_void, // beta
        cusparseDnMatDescr_t, // matC
        cudaDataType, // computeType
        cusparseSpMMAlg_t, // alg
        *mut std::ffi::c_void, // externalBuffer
    ) -> cusparseStatus_t,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CusparseApi {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CusparseApi {}

#[cfg(feature = "cuda")]
static CUSPARSE_API: OnceLock<Option<Arc<CusparseApi>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn get_cusparse() -> Option<Arc<CusparseApi>> {
    CUSPARSE_API
        .get_or_init(|| {
            unsafe {
                let lib = match Library::new("libcusparse.so") {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("eigen-rs [cuda]: Failed to load libcusparse.so: {}", e);
                        return None;
                    }
                };

                let get_sym = |name: &[u8]| -> *const () {
                    match lib.get::<*const ()>(name) {
                        Ok(sym) => *sym,
                        Err(_) => std::ptr::null(),
                    }
                };

                let cusparseCreate_ptr = get_sym(b"cusparseCreate\0");
                let cusparseDestroy_ptr = get_sym(b"cusparseDestroy\0");
                let cusparseCreateCsr_ptr = get_sym(b"cusparseCreateCsr\0");
                let cusparseCreateDnVec_ptr = get_sym(b"cusparseCreateDnVec\0");
                let cusparseCreateDnMat_ptr = get_sym(b"cusparseCreateDnMat\0");
                let cusparseDestroySpMat_ptr = get_sym(b"cusparseDestroySpMat\0");
                let cusparseDestroyDnVec_ptr = get_sym(b"cusparseDestroyDnVec\0");
                let cusparseDestroyDnMat_ptr = get_sym(b"cusparseDestroyDnMat\0");
                let cusparseSpMV_bufferSize_ptr = get_sym(b"cusparseSpMV_bufferSize\0");
                let cusparseSpMV_ptr = get_sym(b"cusparseSpMV\0");
                let cusparseSpMM_bufferSize_ptr = get_sym(b"cusparseSpMM_bufferSize\0");
                let cusparseSpMM_ptr = get_sym(b"cusparseSpMM\0");

                if cusparseCreate_ptr.is_null() || cusparseSpMV_ptr.is_null() {
                    eprintln!("eigen-rs [cuda]: Required cuSPARSE symbols not found.");
                    return None;
                }

                Some(Arc::new(CusparseApi {
                    _lib: lib,
                    cusparseCreate: std::mem::transmute(cusparseCreate_ptr),
                    cusparseDestroy: std::mem::transmute(cusparseDestroy_ptr),
                    cusparseCreateCsr: std::mem::transmute(cusparseCreateCsr_ptr),
                    cusparseCreateDnVec: std::mem::transmute(cusparseCreateDnVec_ptr),
                    cusparseCreateDnMat: std::mem::transmute(cusparseCreateDnMat_ptr),
                    cusparseDestroySpMat: std::mem::transmute(cusparseDestroySpMat_ptr),
                    cusparseDestroyDnVec: std::mem::transmute(cusparseDestroyDnVec_ptr),
                    cusparseDestroyDnMat: std::mem::transmute(cusparseDestroyDnMat_ptr),
                    cusparseSpMV_bufferSize: std::mem::transmute(cusparseSpMV_bufferSize_ptr),
                    cusparseSpMV: std::mem::transmute(cusparseSpMV_ptr),
                    cusparseSpMM_bufferSize: std::mem::transmute(cusparseSpMM_bufferSize_ptr),
                    cusparseSpMM: std::mem::transmute(cusparseSpMM_ptr),
                }))
            }
        })
        .clone()
}

#[cfg(feature = "cuda")]
thread_local! {
    pub static CUSPARSE_HANDLE: Option<cusparseHandle_t> = {
        if let Some(api) = get_cusparse() {
            let mut handle: cusparseHandle_t = std::ptr::null_mut();
            unsafe {
                if (api.cusparseCreate)(&mut handle) == 0 {
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
