//! cuSPARSE wrappers for sparse matrix operations on GPU.

use crate::core::scalar::Scalar;
use crate::core::sparse::cuda_storage::CudaSparseStorage;
use crate::core::tensor::device::cuda::CudaStorage;
#[allow(unused_imports)]
use crate::core::storage::Storage;

#[cfg(feature = "cuda")]
use crate::core::tensor::device::cusparse::{
    get_cusparse, CUSPARSE_HANDLE, CUSPARSE_STATUS_SUCCESS,
};

/// Perform Y = Alpha * A * X + Beta * Y on GPU (SpMV)
pub fn spmv_cuda<T: Scalar>(
    a: &CudaSparseStorage<T>,
    x: &CudaStorage<T>,
    y: &mut CudaStorage<T>,
    alpha: T,
    beta: T,
) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    unsafe {
        let handle_opt = CUSPARSE_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuSPARSE handle not initialized")?;
        let api_opt = get_cusparse();
        let api = api_opt.ok_or("cuSPARSE library not loaded")?;

        let mut mat_a = std::ptr::null_mut();
        let value_type = if std::mem::size_of::<T>() == 8 { 1 } else { 0 };

        let res = (api.cusparseCreateCsr)(
            &mut mat_a,
            a.rows as i64,
            a.cols as i64,
            a.nnz as i64,
            a.row_offsets.as_ptr() as *mut _,
            a.col_indices.as_ptr() as *mut _,
            a.values.as_ptr() as *mut _,
            crate::core::tensor::device::cusparse::CUSPARSE_INDEX_32I,
            crate::core::tensor::device::cusparse::CUSPARSE_INDEX_32I,
            crate::core::tensor::device::cusparse::CUSPARSE_INDEX_BASE_ZERO,
            value_type,
        );
        if res != CUSPARSE_STATUS_SUCCESS {
            return Err("cusparseCreateCsr failed".to_string());
        }

        let mut vec_x = std::ptr::null_mut();
        (api.cusparseCreateDnVec)(
            &mut vec_x,
            a.cols as i64, // Changed from x.rows() to a.cols
            x.as_device_ptr() as *mut _,
            value_type,
        );

        let mut vec_y = std::ptr::null_mut();
        (api.cusparseCreateDnVec)(
            &mut vec_y,
            a.rows as i64, // Changed from y.rows() to a.rows
            y.as_device_ptr() as *mut _,
            value_type,
        );

        let mut buffer_size = 0;
        (api.cusparseSpMV_bufferSize)(
            handle,
            crate::core::tensor::device::cusparse::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const T as *const _,
            mat_a,
            vec_x,
            &beta as *const T as *const _,
            vec_y,
            value_type,
            crate::core::tensor::device::cusparse::CUSPARSE_SPMV_ALG_DEFAULT,
            &mut buffer_size,
        );

        let mut buffer = std::ptr::null_mut();
        if buffer_size > 0 {
            buffer =
                crate::core::tensor::device::cudart::cuda_malloc(buffer_size).unwrap() as *mut _;
        }

        let res = (api.cusparseSpMV)(
            handle,
            crate::core::tensor::device::cusparse::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const T as *const _,
            mat_a,
            vec_x,
            &beta as *const T as *const _,
            vec_y,
            value_type,
            crate::core::tensor::device::cusparse::CUSPARSE_SPMV_ALG_DEFAULT,
            buffer,
        );

        if buffer_size > 0 {
            let _ = crate::core::tensor::device::cudart::cuda_free(buffer);
        }

        (api.cusparseDestroySpMat)(mat_a);
        (api.cusparseDestroyDnVec)(vec_x);
        (api.cusparseDestroyDnVec)(vec_y);

        if res == CUSPARSE_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(format!("cusparseSpMV failed: {:?}", res))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, x, y, alpha, beta);
        Err("CUDA not enabled".to_string())
    }
}

/// Perform C = Alpha * A * B + Beta * C on GPU (SpMM)
pub fn spmm_cuda<T: Scalar>(
    a: &CudaSparseStorage<T>,
    b: &CudaStorage<T>,
    b_rows: usize,
    b_cols: usize,
    c: &mut CudaStorage<T>,
    alpha: T,
    beta: T,
) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    unsafe {
        let handle_opt = CUSPARSE_HANDLE.with(|f| *f);
        let handle = handle_opt.ok_or("cuSPARSE handle not initialized")?;
        let api_opt = get_cusparse();
        let api = api_opt.ok_or("cuSPARSE library not loaded")?;

        let mut mat_a = std::ptr::null_mut();
        let value_type = if std::mem::size_of::<T>() == 8 { 1 } else { 0 };

        let status_a = (api.cusparseCreateCsr)(
            &mut mat_a,
            a.rows as i64,
            a.cols as i64,
            a.nnz as i64,
            a.row_offsets.as_ptr() as *mut _,
            a.col_indices.as_ptr() as *mut _,
            a.values.as_ptr() as *mut _,
            crate::core::tensor::device::cusparse::CUSPARSE_INDEX_32I,
            crate::core::tensor::device::cusparse::CUSPARSE_INDEX_32I,
            crate::core::tensor::device::cusparse::CUSPARSE_INDEX_BASE_ZERO,
            value_type,
        );
        if status_a != 0 {
            return Err("cusparseCreateCsr failed for mat_a".to_string());
        }

        let mut mat_b = std::ptr::null_mut();
        let status_b = (api.cusparseCreateDnMat)(
            &mut mat_b,
            b_rows as i64,
            b_cols as i64,
            b_rows as i64,
            b.as_device_ptr() as *mut _,
            value_type,
            1, // CUSPARSE_ORDER_COL
        );
        if status_b != 0 {
            (api.cusparseDestroySpMat)(mat_a);
            return Err("cusparseCreateDnMat failed for mat_b".to_string());
        }

        let mut mat_c = std::ptr::null_mut();
        let c_rows = a.rows;
        let c_cols = b_cols;
        let status_c = (api.cusparseCreateDnMat)(
            &mut mat_c,
            c_rows as i64,
            c_cols as i64,
            c_rows as i64,
            c.as_device_ptr() as *mut _,
            value_type,
            1, // CUSPARSE_ORDER_COL
        );
        if status_c != 0 {
            (api.cusparseDestroySpMat)(mat_a);
            (api.cusparseDestroyDnMat)(mat_b);
            return Err("cusparseCreateDnMat failed for mat_c".to_string());
        }

        let mut buffer_size = 0;
        let op_a = crate::core::tensor::device::cusparse::CUSPARSE_OPERATION_NON_TRANSPOSE;
        let op_b = crate::core::tensor::device::cusparse::CUSPARSE_OPERATION_NON_TRANSPOSE;
        let alg = crate::core::tensor::device::cusparse::CUSPARSE_SPMM_ALG_DEFAULT;

        let status_size = (api.cusparseSpMM_bufferSize)(
            handle,
            op_a,
            op_b,
            &alpha as *const T as *const _,
            mat_a,
            mat_b,
            &beta as *const T as *const _,
            mat_c,
            value_type,
            alg,
            &mut buffer_size,
        );
        if status_size != 0 {
            (api.cusparseDestroySpMat)(mat_a);
            (api.cusparseDestroyDnMat)(mat_b);
            (api.cusparseDestroyDnMat)(mat_c);
            return Err(format!("cusparseSpMM_bufferSize failed: {}", status_size));
        }

        let mut buffer_ptr = std::ptr::null_mut();
        if buffer_size > 0 {
            buffer_ptr = crate::core::tensor::device::cudart::cuda_malloc(buffer_size)?;
        }

        let status_spmm = (api.cusparseSpMM)(
            handle,
            op_a,
            op_b,
            &alpha as *const T as *const _,
            mat_a,
            mat_b,
            &beta as *const T as *const _,
            mat_c,
            value_type,
            alg,
            buffer_ptr,
        );

        if !buffer_ptr.is_null() {
            crate::core::tensor::device::cudart::cuda_free(buffer_ptr)?;
        }

        (api.cusparseDestroySpMat)(mat_a);
        (api.cusparseDestroyDnMat)(mat_b);
        (api.cusparseDestroyDnMat)(mat_c);

        if status_spmm != 0 {
            return Err(format!("cusparseSpMM failed: {}", status_spmm));
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, c, alpha, beta);
        Err("CUDA not enabled".to_string())
    }
}
