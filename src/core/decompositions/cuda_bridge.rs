//! CUDA Hardware Acceleration Bridge for Decompositions.
//!
//! This module securely wraps the NVIDIA `cuBLAS` and `cuSOLVER` libraries using dynamic module loading
//! (`libloading`) to avoid strict static linking dependencies. Key features include:
//! - **CudaDecompositionExt**: A trait implemented on standard Rust matrices enabling fallback-safe GPU operations.
//! - **High-Performance Decompositions**: Accelerates LLT (Cholesky), LU, QR, and SVD factorizations for massive systems.
//! - **Asynchronous Portability**: Safely operates on GitHub Actions and CI environments without native CUDA toolkits installed.

#[cfg(feature = "cuda")]
use crate::core::matrix::Matrix;
#[cfg(feature = "cuda")]
use crate::core::scalar::Scalar;
#[cfg(feature = "cuda")]
use crate::core::storage::Storage;
#[cfg(feature = "cuda")]
use crate::core::tensor::device::cuda::CudaStorage;
#[cfg(feature = "cuda")]
use crate::core::tensor::device::DeviceStorage;

#[cfg(feature = "cuda")]
pub trait CudaDecompositionExt<T: Scalar, S: Storage<T>> {
    /// Attempts to compute the LU Decomposition on the GPU and returns
    /// the factored matrix along with the permutation vector `p` and determinant `det_p`.
    fn try_lu_cuda(
        &self,
    ) -> Result<
        Option<(
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
            Vec<usize>,
            T,
        )>,
        String,
    >;

    /// Attempts to compute the LLT (Cholesky) Decomposition on the GPU.
    /// Returns the lower triangular matrix L.
    fn try_llt_cuda(
        &self,
    ) -> Result<Option<Matrix<T, crate::core::storage::DynamicStorage<T>>>, String>;

    /// Attempts to compute the SVD Decomposition on the GPU.
    /// Returns the U, S, V components. Note: V corresponds to the right singular vectors natively without Hermitian conjugation.
    fn try_svd_cuda(
        &self,
    ) -> Result<
        Option<(
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
        )>,
        String,
    >;

    /// Attempts to compute the QR Decomposition on the GPU.
    /// Returns the matrix containing R (upper) and Householder vectors (strict lower), and the `h_coeffs` (tau vector).
    fn try_qr_cuda(
        &self,
    ) -> Result<Option<(Matrix<T, crate::core::storage::DynamicStorage<T>>, Vec<T>)>, String>;
}

#[cfg(feature = "cuda")]
impl<T: Scalar, S: Storage<T>> CudaDecompositionExt<T, S> for Matrix<T, S> {
    fn try_lu_cuda(
        &self,
    ) -> Result<
        Option<(
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
            Vec<usize>,
            T,
        )>,
        String,
    > {
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

        if !is_f32 && !is_f64 {
            return Ok(None);
        }

        let m = self.rows();
        let n = self.cols();

        let lda = m as i32;

        let handled_cusolver = crate::core::tensor::device::cusolver::CUSOLVER_HANDLE.with(|h| {
            if let Some(handle) = h {
                if let Some(api) = crate::core::tensor::device::cusolver::get_cusolver() {
                    unsafe {
                        let mut lu =
                            Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(
                                m, n,
                            )?;
                        lu.assign(self)?;

                        let min_dim = std::cmp::min(m, n);
                        let mut lwork: i32 = 0;

                        if is_f32 {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 4)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(
                                d_a,
                                lu.storage().data().as_ptr() as *const _,
                                m * n * 4,
                            )?;

                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;
                            let d_ipiv =
                                crate::core::tensor::device::cudart::cuda_malloc(min_dim * 4)?;

                            let p_a = d_a as *mut f32;

                            let status_ws = (api.cusolverDnSgetrf_bufferSize)(
                                *handle, m as i32, n as i32, p_a, lda, &mut lwork,
                            );

                            if status_ws
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                return Err(format!(
                                    "cusolverDnSgetrf_bufferSize failed: {}",
                                    status_ws
                                ));
                            }

                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(
                                lwork as usize * 4,
                            )?;

                            let p_work = d_work as *mut f32;
                            let p_ipiv = d_ipiv as *mut i32;
                            let p_info = d_info as *mut i32;

                            let status = (api.cusolverDnSgetrf)(
                                *handle, m as i32, n as i32, p_a, lda, p_work, p_ipiv, p_info,
                            );

                            if status
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                return Err(format!("cusolverDnSgetrf failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;

                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_info.as_mut_ptr() as *mut _,
                                d_info,
                                4,
                            )?;

                            if h_info[0] < 0 {
                                return Err(format!(
                                    "cuSOLVER getrf returned invalid arg at {}",
                                    -h_info[0]
                                ));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                lu.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_a,
                                m * n * 4,
                            )?;

                            let mut h_ipiv = vec![0i32; min_dim];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_ipiv.as_mut_ptr() as *mut _,
                                d_ipiv,
                                min_dim * 4,
                            )?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_ipiv)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;

                            return Ok(Some((lu, h_ipiv)));
                        } else {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 8)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(
                                d_a,
                                lu.storage().data().as_ptr() as *const _,
                                m * n * 8,
                            )?;

                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;
                            let d_ipiv =
                                crate::core::tensor::device::cudart::cuda_malloc(min_dim * 4)?;

                            let p_a = d_a as *mut f64;

                            let status_ws = (api.cusolverDnDgetrf_bufferSize)(
                                *handle, m as i32, n as i32, p_a, lda, &mut lwork,
                            );

                            if status_ws
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                return Err(format!(
                                    "cusolverDnDgetrf_bufferSize failed: {}",
                                    status_ws
                                ));
                            }

                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(
                                lwork as usize * 8,
                            )?;

                            let p_work = d_work as *mut f64;
                            let p_ipiv = d_ipiv as *mut i32;
                            let p_info = d_info as *mut i32;

                            let status = (api.cusolverDnDgetrf)(
                                *handle, m as i32, n as i32, p_a, lda, p_work, p_ipiv, p_info,
                            );

                            if status
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                return Err(format!("cusolverDnDgetrf failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;

                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_info.as_mut_ptr() as *mut _,
                                d_info,
                                4,
                            )?;

                            if h_info[0] < 0 {
                                return Err(format!(
                                    "cuSOLVER getrf returned invalid arg at {}",
                                    -h_info[0]
                                ));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                lu.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_a,
                                m * n * 8,
                            )?;

                            let mut h_ipiv = vec![0i32; min_dim];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_ipiv.as_mut_ptr() as *mut _,
                                d_ipiv,
                                min_dim * 4,
                            )?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_ipiv)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;

                            return Ok(Some((lu, h_ipiv)));
                        }
                    }
                }
            }
            Ok(None)
        });

        let (lu, h_ipiv) = match handled_cusolver? {
            Some(data) => data,
            None => return Ok(None),
        };

        // Convert 1-based cuSOLVER pivots to 0-based rust permutations
        let mut p: Vec<usize> = (0..m).collect();
        let mut det_p = T::from_usize(1);
        let neg_one = T::from_usize(0) - T::from_usize(1);

        let min_dim = std::cmp::min(m, n);
        for i in 0..min_dim {
            let pivot_idx = (h_ipiv[i] - 1) as usize; // cuSOLVER uses 1-based indexing for Fortran
            if i != pivot_idx {
                p.swap(i, pivot_idx);
                det_p = det_p * neg_one;
            }
        }

        Ok(Some((lu, p, det_p)))
    }

    fn try_llt_cuda(
        &self,
    ) -> Result<Option<Matrix<T, crate::core::storage::DynamicStorage<T>>>, String> {
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

        if !is_f32 && !is_f64 {
            return Ok(None);
        }

        let m = self.rows();
        let n = self.cols();

        if m != n {
            return Err("LLT requires a square matrix".to_string());
        }

        // Return instantly for tiny matrices or if GPU overhead isn't worth it on small sizes.
        // C++ Eigen does this too. (E.g < 32)
        if m < 32 {
            return Ok(None);
        }

        let lda = m as i32;
        let fill_mode = crate::core::tensor::device::cublas::CUBLAS_FILL_MODE_LOWER;

        let handled_cusolver = crate::core::tensor::device::cusolver::CUSOLVER_HANDLE.with(|h| {
            if let Some(handle) = h {
                if let Some(api) = crate::core::tensor::device::cusolver::get_cusolver() {
                    unsafe {
                        let mut l = Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(m, n)?;
                        l.assign(self)?;
                        
                        let mut lwork: i32 = 0;

                        if is_f32 {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 4)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(d_a, l.storage().data().as_ptr() as *const _, m * n * 4)?;
                            
                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;
                            
                            let p_a = d_a as *mut f32;
                            
                            let status_ws = (api.cusolverDnSpotrf_bufferSize)(
                                *handle, fill_mode, n as i32, p_a, lda, &mut lwork
                            );
                            
                            if status_ws != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                return Err(format!("cusolverDnSpotrf_bufferSize failed: {}", status_ws));
                            }
                            
                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(lwork as usize * 4)?;
                            let p_work = d_work as *mut f32;
                            let p_info = d_info as *mut i32;

                            let status = (api.cusolverDnSpotrf)(
                                *handle, fill_mode, n as i32, p_a, lda, p_work, lwork, p_info
                            );
                            
                            if status != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cusolverDnSpotrf failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;
                            
                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(h_info.as_mut_ptr() as *mut _, d_info, 4)?;
                            
                            if h_info[0] > 0 {
                                // Matrix is not positive definite
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cuSOLVER potrf (LLT) matrix is not positive definite. Minor: {}", h_info[0]));
                            } else if h_info[0] < 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cuSOLVER potrf (LLT) returned invalid arg at {}", -h_info[0]));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(l.storage_mut().data_mut().as_mut_ptr() as *mut _, d_a, m * n * 4)?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;

                            return Ok(Some(l));
                        } else {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 8)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(d_a, l.storage().data().as_ptr() as *const _, m * n * 8)?;
                            
                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;
                            
                            let p_a = d_a as *mut f64;
                            
                            let status_ws = (api.cusolverDnDpotrf_bufferSize)(
                                *handle, fill_mode, n as i32, p_a, lda, &mut lwork
                            );
                            
                            if status_ws != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                return Err(format!("cusolverDnDpotrf_bufferSize failed: {}", status_ws));
                            }
                            
                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(lwork as usize * 8)?;
                            let p_work = d_work as *mut f64;
                            let p_info = d_info as *mut i32;

                            let status = (api.cusolverDnDpotrf)(
                                *handle, fill_mode, n as i32, p_a, lda, p_work, lwork, p_info
                            );
                            
                            if status != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cusolverDnDpotrf failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;
                            
                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(h_info.as_mut_ptr() as *mut _, d_info, 4)?;
                            
                            if h_info[0] > 0 {
                                // Matrix is not positive definite
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cuSOLVER potrf (LLT) matrix is not positive definite. Minor: {}", h_info[0]));
                            } else if h_info[0] < 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cuSOLVER potrf (LLT) returned invalid arg at {}", -h_info[0]));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(l.storage_mut().data_mut().as_mut_ptr() as *mut _, d_a, m * n * 8)?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;

                            return Ok(Some(l));
                        }
                    }
                }
            }
            Ok(None)
        });

        let mut l = match handled_cusolver? {
            Some(data) => data,
            None => return Ok(None),
        };

        Ok(Some(l))
    }

    fn try_svd_cuda(
        &self,
    ) -> Result<
        Option<(
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
            Matrix<T, crate::core::storage::DynamicStorage<T>>,
        )>,
        String,
    > {
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

        if !is_f32 && !is_f64 {
            return Ok(None);
        }

        let m = self.rows();
        let n = self.cols();

        // Return instantly for tiny matrices or if GPU overhead isn't worth it on small sizes.
        if std::cmp::min(m, n) < 32 {
            return Ok(None);
        }

        let lda = m as i32;
        let ldu = m as i32;
        let ldv = n as i32;
        let jobu = b'A' as i8; // All M columns of U
        let jobvt = b'A' as i8; // All N rows of V^T (translates to N columns of V)

        let handled_cusolver = crate::core::tensor::device::cusolver::CUSOLVER_HANDLE.with(|h| {
            if let Some(handle) = h {
                if let Some(api) = crate::core::tensor::device::cusolver::get_cusolver() {
                    unsafe {
                        let mut a =
                            Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(
                                m, n,
                            )?;
                        a.assign(self)?;

                        let min_dim = std::cmp::min(m, n);
                        let mut s =
                            Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(
                                min_dim, 1,
                            )?;
                        let mut u =
                            Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(
                                m, m,
                            )?;
                        let mut vt =
                            Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(
                                n, n,
                            )?;

                        let mut lwork: i32 = 0;

                        if is_f32 {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 4)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(
                                d_a,
                                a.storage().data().as_ptr() as *const _,
                                m * n * 4,
                            )?;

                            let d_s =
                                crate::core::tensor::device::cudart::cuda_malloc(min_dim * 4)?;
                            let d_u = crate::core::tensor::device::cudart::cuda_malloc(m * m * 4)?;
                            let d_vt = crate::core::tensor::device::cudart::cuda_malloc(n * n * 4)?;
                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;

                            let p_a = d_a as *mut f32;
                            let p_s = d_s as *mut f32;
                            let p_u = d_u as *mut f32;
                            let p_vt = d_vt as *mut f32;

                            let status_ws = (api.cusolverDnSgesvd_bufferSize)(
                                *handle, m as i32, n as i32, &mut lwork,
                            );

                            if status_ws
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                return Err(format!(
                                    "cusolverDnSgesvd_bufferSize failed: {}",
                                    status_ws
                                ));
                            }

                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(
                                lwork as usize * 4,
                            )?;
                            let p_work = d_work as *mut f32;
                            let p_info = d_info as *mut i32;

                            // sgesvd expects devInfo
                            // Note: `rwork` (real working space) is generally NULL for real matrices,
                            // but FFI expects the signature: ..., p_work, lwork, p_rwork, p_info
                            // For sgesvd, rwork is historically an f32 buffer of 5*min(m,n), but our FFI binding has it as `*mut f32`
                            // In cuSOLVER API for real SVD, rwork is technically *mut f32.
                            let d_rwork =
                                crate::core::tensor::device::cudart::cuda_malloc(5 * min_dim * 4)?;
                            let p_rwork = d_rwork as *mut f32;
                            let status = (api.cusolverDnSgesvd)(
                                *handle, jobu, jobvt, m as i32, n as i32, p_a, lda, p_s, p_u, ldu,
                                p_vt, ldv, p_work, lwork, p_rwork, p_info,
                            );

                            if status
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                crate::core::tensor::device::cudart::cuda_free(d_rwork)?;
                                return Err(format!("cusolverDnSgesvd failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;

                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_info.as_mut_ptr() as *mut _,
                                d_info,
                                4,
                            )?;

                            if h_info[0] > 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                crate::core::tensor::device::cudart::cuda_free(d_rwork)?;
                                return Err(format!(
                                    "cuSOLVER gesvd (SVD) did not converge. Minor: {}",
                                    h_info[0]
                                ));
                            } else if h_info[0] < 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                crate::core::tensor::device::cudart::cuda_free(d_rwork)?;
                                return Err(format!(
                                    "cuSOLVER gesvd (SVD) returned invalid arg at {}",
                                    -h_info[0]
                                ));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                s.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_s,
                                min_dim * 4,
                            )?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                u.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_u,
                                m * m * 4,
                            )?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                vt.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_vt,
                                n * n * 4,
                            )?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_s)?;
                            crate::core::tensor::device::cudart::cuda_free(d_u)?;
                            crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;
                            crate::core::tensor::device::cudart::cuda_free(d_rwork)?;

                            return Ok(Some((u, s, vt)));
                        } else {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 8)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(
                                d_a,
                                a.storage().data().as_ptr() as *const _,
                                m * n * 8,
                            )?;

                            let d_s =
                                crate::core::tensor::device::cudart::cuda_malloc(min_dim * 8)?;
                            let d_u = crate::core::tensor::device::cudart::cuda_malloc(m * m * 8)?;
                            let d_vt = crate::core::tensor::device::cudart::cuda_malloc(n * n * 8)?;
                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;

                            let p_a = d_a as *mut f64;
                            let p_s = d_s as *mut f64;
                            let p_u = d_u as *mut f64;
                            let p_vt = d_vt as *mut f64;

                            let status_ws = (api.cusolverDnDgesvd_bufferSize)(
                                *handle, m as i32, n as i32, &mut lwork,
                            );

                            if status_ws
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                return Err(format!(
                                    "cusolverDnDgesvd_bufferSize failed: {}",
                                    status_ws
                                ));
                            }

                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(
                                lwork as usize * 8,
                            )?;
                            let p_work = d_work as *mut f64;
                            let p_info = d_info as *mut i32;

                            let d_rwork =
                                crate::core::tensor::device::cudart::cuda_malloc(5 * min_dim * 8)?;
                            let p_rwork = d_rwork as *mut f64;
                            let status = (api.cusolverDnDgesvd)(
                                *handle, jobu, jobvt, m as i32, n as i32, p_a, lda, p_s, p_u, ldu,
                                p_vt, ldv, p_work, lwork, p_rwork, p_info,
                            );

                            if status
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                crate::core::tensor::device::cudart::cuda_free(d_rwork)?;
                                return Err(format!("cusolverDnDgesvd failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;

                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_info.as_mut_ptr() as *mut _,
                                d_info,
                                4,
                            )?;

                            if h_info[0] > 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                crate::core::tensor::device::cudart::cuda_free(d_rwork)?;
                                return Err(format!(
                                    "cuSOLVER gesvd (SVD) did not converge. Minor: {}",
                                    h_info[0]
                                ));
                            } else if h_info[0] < 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_s)?;
                                crate::core::tensor::device::cudart::cuda_free(d_u)?;
                                crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                crate::core::tensor::device::cudart::cuda_free(d_rwork)?;
                                return Err(format!(
                                    "cuSOLVER gesvd (SVD) returned invalid arg at {}",
                                    -h_info[0]
                                ));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                s.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_s,
                                min_dim * 8,
                            )?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                u.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_u,
                                m * m * 8,
                            )?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                vt.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_vt,
                                n * n * 8,
                            )?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_s)?;
                            crate::core::tensor::device::cudart::cuda_free(d_u)?;
                            crate::core::tensor::device::cudart::cuda_free(d_vt)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;
                            crate::core::tensor::device::cudart::cuda_free(d_rwork)?;

                            return Ok(Some((u, s, vt)));
                        }
                    }
                }
            }
            Ok(None)
        });

        match handled_cusolver? {
            Some(data) => {
                let (u, s, vt) = data;
                // V is returned as V^H (V^T for real matrices). We need to transpose it back to V.
                let mut v =
                    Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(n, n)?;
                for i in 0..n {
                    for j in 0..n {
                        *v.get_mut(i, j).unwrap() = *vt.get(j, i).unwrap();
                    }
                }
                Ok(Some((u, s, v)))
            }
            None => return Ok(None),
        }
    }

    fn try_qr_cuda(
        &self,
    ) -> Result<Option<(Matrix<T, crate::core::storage::DynamicStorage<T>>, Vec<T>)>, String> {
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
        let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

        if !is_f32 && !is_f64 {
            return Ok(None);
        }

        let m = self.rows();
        let n = self.cols();

        // Return instantly for tiny matrices
        if std::cmp::min(m, n) < 32 {
            return Ok(None);
        }

        let lda = m as i32;
        let min_dim = std::cmp::min(m, n);

        let handled_cusolver = crate::core::tensor::device::cusolver::CUSOLVER_HANDLE.with(|h| {
            if let Some(handle) = h {
                if let Some(api) = crate::core::tensor::device::cusolver::get_cusolver() {
                    unsafe {
                        let mut qr =
                            Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(
                                m, n,
                            )?;
                        qr.assign(self)?;

                        let mut h_coeffs = vec![T::default(); min_dim];
                        let mut lwork: i32 = 0;

                        if is_f32 {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 4)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(
                                d_a,
                                qr.storage().data().as_ptr() as *const _,
                                m * n * 4,
                            )?;

                            let d_tau =
                                crate::core::tensor::device::cudart::cuda_malloc(min_dim * 4)?;
                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;

                            let p_a = d_a as *mut f32;
                            let p_tau = d_tau as *mut f32;

                            let status_ws = (api.cusolverDnSgeqrf_bufferSize)(
                                *handle, m as i32, n as i32, p_a, lda, &mut lwork,
                            );

                            if status_ws
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                return Err(format!(
                                    "cusolverDnSgeqrf_bufferSize failed: {}",
                                    status_ws
                                ));
                            }

                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(
                                lwork as usize * 4,
                            )?;
                            let p_work = d_work as *mut f32;
                            let p_info = d_info as *mut i32;

                            let status = (api.cusolverDnSgeqrf)(
                                *handle, m as i32, n as i32, p_a, lda, p_tau, p_work, lwork, p_info,
                            );

                            if status
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cusolverDnSgeqrf failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;

                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_info.as_mut_ptr() as *mut _,
                                d_info,
                                4,
                            )?;

                            if h_info[0] < 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!(
                                    "cuSOLVER geqrf (QR) returned invalid arg at {}",
                                    -h_info[0]
                                ));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                qr.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_a,
                                m * n * 4,
                            )?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_coeffs.as_mut_ptr() as *mut _,
                                d_tau,
                                min_dim * 4,
                            )?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;

                            return Ok(Some((qr, h_coeffs)));
                        } else {
                            let d_a = crate::core::tensor::device::cudart::cuda_malloc(m * n * 8)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_h2d(
                                d_a,
                                qr.storage().data().as_ptr() as *const _,
                                m * n * 8,
                            )?;

                            let d_tau =
                                crate::core::tensor::device::cudart::cuda_malloc(min_dim * 8)?;
                            let d_info = crate::core::tensor::device::cudart::cuda_malloc(4)?;

                            let p_a = d_a as *mut f64;
                            let p_tau = d_tau as *mut f64;

                            let status_ws = (api.cusolverDnDgeqrf_bufferSize)(
                                *handle, m as i32, n as i32, p_a, lda, &mut lwork,
                            );

                            if status_ws
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                return Err(format!(
                                    "cusolverDnDgeqrf_bufferSize failed: {}",
                                    status_ws
                                ));
                            }

                            let d_work = crate::core::tensor::device::cudart::cuda_malloc(
                                lwork as usize * 8,
                            )?;
                            let p_work = d_work as *mut f64;
                            let p_info = d_info as *mut i32;

                            let status = (api.cusolverDnDgeqrf)(
                                *handle, m as i32, n as i32, p_a, lda, p_tau, p_work, lwork, p_info,
                            );

                            if status
                                != crate::core::tensor::device::cusolver::CUSOLVER_STATUS_SUCCESS
                            {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!("cusolverDnDgeqrf failed: {}", status));
                            }

                            crate::core::tensor::device::cudart::cuda_device_synchronize()?;

                            let mut h_info = vec![0i32; 1];
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_info.as_mut_ptr() as *mut _,
                                d_info,
                                4,
                            )?;

                            if h_info[0] < 0 {
                                crate::core::tensor::device::cudart::cuda_free(d_a)?;
                                crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                                crate::core::tensor::device::cudart::cuda_free(d_info)?;
                                crate::core::tensor::device::cudart::cuda_free(d_work)?;
                                return Err(format!(
                                    "cuSOLVER geqrf (QR) returned invalid arg at {}",
                                    -h_info[0]
                                ));
                            }

                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                qr.storage_mut().data_mut().as_mut_ptr() as *mut _,
                                d_a,
                                m * n * 8,
                            )?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2h(
                                h_coeffs.as_mut_ptr() as *mut _,
                                d_tau,
                                min_dim * 8,
                            )?;

                            crate::core::tensor::device::cudart::cuda_free(d_a)?;
                            crate::core::tensor::device::cudart::cuda_free(d_tau)?;
                            crate::core::tensor::device::cudart::cuda_free(d_info)?;
                            crate::core::tensor::device::cudart::cuda_free(d_work)?;

                            return Ok(Some((qr, h_coeffs)));
                        }
                    }
                }
            }
            Ok(None)
        });

        match handled_cusolver? {
            Some(data) => Ok(Some(data)),
            None => return Ok(None),
        }
    }
}
