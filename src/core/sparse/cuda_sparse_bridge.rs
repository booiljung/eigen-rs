//! CUDA bridging extensions for SparseMatrix

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::cuda_ops::{spmm_cuda, spmv_cuda};
use crate::core::sparse::cuda_storage::CudaSparseStorage;
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::storage::DynamicStorage;
use crate::core::storage::Storage;
use crate::core::tensor::device::cuda::CudaDevice;
use crate::core::tensor::device::Device;
use crate::core::tensor::device::cuda::CudaStorage;
use crate::core::tensor::device::DeviceStorage;

pub trait CudaSparseExt<T: Scalar> {
    fn try_mul_dense_cuda<S: Storage<T>, R: Storage<T>>(
        &self,
        rhs: &Matrix<T, S>,
        res: &mut Matrix<T, R>,
    ) -> Result<Option<()>, String>;

    fn try_bicgstab_cuda<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Option<Matrix<T, DynamicStorage<T>>>, String>;
}

impl<T: Scalar> CudaSparseExt<T> for SparseMatrix<T> {
    fn try_mul_dense_cuda<S: Storage<T>, R: Storage<T>>(
        &self,
        rhs: &Matrix<T, S>,
        res: &mut Matrix<T, R>,
    ) -> Result<Option<()>, String> {
        #[cfg(feature = "cuda")]
        {
            if !crate::core::tensor::device::cuda::is_cuda_device_active() {
                return Ok(None);
            }

            let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
            let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

            if !is_f32 && !is_f64 {
                return Ok(None);
            }

            if self.order() != crate::core::sparse::sparse_matrix::StorageOrder::RowMajor {
                return Ok(None); // cusparseCreateCsr requires RowMajor formats easily
            }

            if self.rows() < 32 {
                return Ok(None); // GPU initialization/transfer overhead dominates
            }

            let mut a_gpu = CudaSparseStorage::<T>::new(self.rows(), self.cols(), self.non_zeros())?;
            
            let inner_i32: Vec<i32> = self.inner_indices().iter().map(|&x| x as i32).collect();
            let outer_i32: Vec<i32> = self.outer_starts().iter().map(|&x| x as i32).collect();
            a_gpu.copy_from_host(self.values(), &inner_i32, &outer_i32)?;

            let n_elem = rhs.rows() * rhs.cols();
            let mut x_gpu = CudaStorage::<T>::new(n_elem)?;
            if rhs.storage().data().len() == n_elem {
                x_gpu.copy_from_host(rhs.storage().data())?;
            } else {
                let mut rhs_vec = vec![T::default(); n_elem];
                for j in 0..rhs.cols() {
                    for i in 0..rhs.rows() {
                        rhs_vec[j * rhs.rows() + i] = *rhs.get(i, j).unwrap();
                    }
                }
                x_gpu.copy_from_host(&rhs_vec)?;
            }

            let res_elem = res.rows() * res.cols();
            let mut y_gpu = CudaStorage::<T>::new(res_elem)?;
            if res.storage().data().len() == res_elem {
                y_gpu.copy_from_host(res.storage().data())?;
            } else {
                let mut y_vec = vec![T::default(); res_elem];
                for j in 0..res.cols() {
                    for i in 0..res.rows() {
                        y_vec[j * res.rows() + i] = *res.get(i, j).unwrap();
                    }
                }
                y_gpu.copy_from_host(&y_vec)?;
            }

            if rhs.cols() == 1 {
                spmv_cuda(&a_gpu, &x_gpu, &mut y_gpu, T::from_f64(1.0), T::from_f64(0.0))?;
            } else {
                spmm_cuda(&a_gpu, &x_gpu, rhs.rows(), rhs.cols(), &mut y_gpu, T::from_f64(1.0), T::from_f64(0.0))?;
            }

            if res.storage_mut().data_mut().len() == res_elem {
                y_gpu.copy_to_host(res.storage_mut().data_mut())?;
            } else {
                let mut y_vec = vec![T::default(); res_elem];
                y_gpu.copy_to_host(&mut y_vec)?;
                for j in 0..res.cols() {
                    for i in 0..res.rows() {
                        *res.get_mut(i, j).unwrap() = y_vec[j * res.rows() + i];
                    }
                }
            }

            return Ok(Some(()));
        }

        #[cfg(not(feature = "cuda"))]
        Ok(None)
    }

    fn try_bicgstab_cuda<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Option<Matrix<T, DynamicStorage<T>>>, String> {
        #[cfg(feature = "cuda")]
        {
            if !crate::core::tensor::device::cuda::is_cuda_device_active() {
                return Ok(None);
            }

            let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
            let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();

            if !is_f32 && !is_f64 {
                return Ok(None);
            }

            if self.order() != crate::core::sparse::sparse_matrix::StorageOrder::RowMajor {
                return Ok(None);
            }

            let n = self.rows();
            if n < 32 {
                return Ok(None); // GPU initialization/transfer overhead dominates
            }

            // Setup A on GPU
            let mut a_gpu = CudaSparseStorage::<T>::new(self.rows(), self.cols(), self.non_zeros())?;
            let inner_i32: Vec<i32> = self.inner_indices().iter().map(|&x| x as i32).collect();
            let outer_i32: Vec<i32> = self.outer_starts().iter().map(|&x| x as i32).collect();
            a_gpu.copy_from_host(self.values(), &inner_i32, &outer_i32)?;

            let cuda_dev = CudaDevice::default();
            let mut x_out = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;

            for k in 0..b.cols() {
                // b_vec
                let mut b_vec = vec![T::default(); n];
                for i in 0..n {
                    b_vec[i] = *b.get(i, k).unwrap();
                }

                let mut x_gpu = CudaStorage::<T>::new(n)?;
                let mut p_gpu = CudaStorage::<T>::new(n)?;
                let mut r_gpu = CudaStorage::<T>::new(n)?;
                let mut r_tilde_gpu = CudaStorage::<T>::new(n)?;
                let mut v_gpu = CudaStorage::<T>::new(n)?;
                let mut s_gpu = CudaStorage::<T>::new(n)?;
                let mut t_gpu = CudaStorage::<T>::new(n)?;

                let zeros = vec![T::default(); n];
                x_gpu.copy_from_host(&zeros)?;
                
                // r = b
                r_gpu.copy_from_host(&b_vec)?;
                // r_tilde = r
                r_tilde_gpu.copy_from_host(&b_vec)?;

                let b_norm = cuda_dev.norm(&r_gpu)?.to_f64();
                if b_norm < 1e-16 {
                    let mut x_col = vec![T::default(); n];
                    x_gpu.copy_to_host(&mut x_col)?;
                    for i in 0..n {
                        *x_out.get_mut(i, k).unwrap() = x_col[i];
                    }
                    continue; // x remains 0
                }

                let mut rho = T::from_f64(1.0);
                let mut alpha = T::from_f64(1.0);
                let mut omega = T::from_f64(1.0);

                unsafe {
                    for iter in 0..max_iterations {
                        let rho_next = cuda_dev.dot(&r_tilde_gpu, &r_gpu)?;
                        if rho_next.abs().to_f64() < 1e-20 {
                            return Err("BiCGSTAB stalled (rho)".to_string());
                        }

                        if iter == 0 {
                            crate::core::tensor::device::cudart::cuda_memcpy_d2d(
                                p_gpu.as_device_ptr(),
                                r_gpu.as_device_ptr() as *const _,
                                n * std::mem::size_of::<T>(),
                            )?;
                        } else {
                            let beta = (rho_next / rho) * (alpha / omega);
                            let mut t1 = CudaStorage::<T>::new(n)?;
                            crate::core::tensor::device::cudart::cuda_memcpy_d2d(t1.as_device_ptr(), p_gpu.as_device_ptr() as *const _, n * std::mem::size_of::<T>())?;
                            
                            let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
                            let handle_opt = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f);
                            let handle = handle_opt.unwrap();
                            let api = crate::core::tensor::device::cublas::get_cublas().unwrap();

                            if is_f32 {
                                let minus_omega = -omega.to_f64() as f32;
                                (api.cublasSaxpy_v2)(handle, n as i32, &minus_omega as *const f32, v_gpu.as_device_ptr() as *const f32, 1, t1.as_device_ptr() as *mut f32, 1);
                            } else {
                                let minus_omega = -omega.to_f64() as f64;
                                (api.cublasDaxpy_v2)(handle, n as i32, &minus_omega as *const f64, v_gpu.as_device_ptr() as *const f64, 1, t1.as_device_ptr() as *mut f64, 1);
                            }

                            // p = r + beta * t1
                            crate::core::tensor::device::cudart::cuda_memcpy_d2d(p_gpu.as_device_ptr(), r_gpu.as_device_ptr() as *const _, n * std::mem::size_of::<T>())?;
                            if is_f32 {
                                let b_f32 = beta.to_f64() as f32;
                                (api.cublasSaxpy_v2)(handle, n as i32, &b_f32 as *const f32, t1.as_device_ptr() as *const f32, 1, p_gpu.as_device_ptr() as *mut f32, 1);
                            } else {
                                let b_f64 = beta.to_f64() as f64;
                                (api.cublasDaxpy_v2)(handle, n as i32, &b_f64 as *const f64, t1.as_device_ptr() as *const f64, 1, p_gpu.as_device_ptr() as *mut f64, 1);
                            }
                        }

                        let p_hat = &p_gpu;

                        spmv_cuda(&a_gpu, p_hat, &mut v_gpu, T::from_f64(1.0), T::from_f64(0.0))?;

                        let rvt = cuda_dev.dot(&r_tilde_gpu, &v_gpu)?;
                        alpha = rho_next / rvt;

                        crate::core::tensor::device::cudart::cuda_memcpy_d2d(s_gpu.as_device_ptr(), r_gpu.as_device_ptr() as *const _, n * std::mem::size_of::<T>())?;
                        let api = crate::core::tensor::device::cublas::get_cublas().unwrap();
                        let handle = crate::core::tensor::device::cublas::CUBLAS_HANDLE.with(|f| *f).unwrap();
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                            let minus_alpha = -alpha.to_f64() as f32;
                            (api.cublasSaxpy_v2)(handle, n as i32, &minus_alpha as *const f32, v_gpu.as_device_ptr() as *const f32, 1, s_gpu.as_device_ptr() as *mut f32, 1);
                        } else {
                            let minus_alpha = -alpha.to_f64() as f64;
                            (api.cublasDaxpy_v2)(handle, n as i32, &minus_alpha as *const f64, v_gpu.as_device_ptr() as *const f64, 1, s_gpu.as_device_ptr() as *mut f64, 1);
                        }

                        if cuda_dev.norm(&s_gpu)?.to_f64() / b_norm < tolerance {
                            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                                let a_f32 = alpha.to_f64() as f32;
                                (api.cublasSaxpy_v2)(handle, n as i32, &a_f32 as *const f32, p_hat.as_device_ptr() as *const f32, 1, x_gpu.as_device_ptr() as *mut f32, 1);
                            } else {
                                let a_f64 = alpha.to_f64() as f64;
                                (api.cublasDaxpy_v2)(handle, n as i32, &a_f64 as *const f64, p_hat.as_device_ptr() as *const f64, 1, x_gpu.as_device_ptr() as *mut f64, 1);
                            }
                            break;
                        }

                        let s_hat = &s_gpu;

                        spmv_cuda(&a_gpu, s_hat, &mut t_gpu, T::from_f64(1.0), T::from_f64(0.0))?;

                        omega = cuda_dev.dot(&t_gpu, &s_gpu)? / cuda_dev.dot(&t_gpu, &t_gpu)?;

                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                            let a_f32 = alpha.to_f64() as f32;
                            (api.cublasSaxpy_v2)(handle, n as i32, &a_f32 as *const f32, p_hat.as_device_ptr() as *const f32, 1, x_gpu.as_device_ptr() as *mut f32, 1);
                            let w_f32 = omega.to_f64() as f32;
                            (api.cublasSaxpy_v2)(handle, n as i32, &w_f32 as *const f32, s_hat.as_device_ptr() as *const f32, 1, x_gpu.as_device_ptr() as *mut f32, 1);
                        } else {
                            let a_f64 = alpha.to_f64() as f64;
                            (api.cublasDaxpy_v2)(handle, n as i32, &a_f64 as *const f64, p_hat.as_device_ptr() as *const f64, 1, x_gpu.as_device_ptr() as *mut f64, 1);
                            let w_f64 = omega.to_f64() as f64;
                            (api.cublasDaxpy_v2)(handle, n as i32, &w_f64 as *const f64, s_hat.as_device_ptr() as *const f64, 1, x_gpu.as_device_ptr() as *mut f64, 1);
                        }

                        crate::core::tensor::device::cudart::cuda_memcpy_d2d(r_gpu.as_device_ptr(), s_gpu.as_device_ptr() as *const _, n * std::mem::size_of::<T>())?;
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                            let minus_omega = -omega.to_f64() as f32;
                            (api.cublasSaxpy_v2)(handle, n as i32, &minus_omega as *const f32, t_gpu.as_device_ptr() as *const f32, 1, r_gpu.as_device_ptr() as *mut f32, 1);
                        } else {
                            let minus_omega = -omega.to_f64() as f64;
                            (api.cublasDaxpy_v2)(handle, n as i32, &minus_omega as *const f64, t_gpu.as_device_ptr() as *const f64, 1, r_gpu.as_device_ptr() as *mut f64, 1);
                        }

                        rho = rho_next;

                        if cuda_dev.norm(&r_gpu)?.to_f64() / b_norm < tolerance {
                            break;
                        }

                        if omega.abs().to_f64() < 1e-20 {
                            return Err("BiCGSTAB stalled (omega)".to_string());
                        }
                    }
                }

                let mut x_col = vec![T::default(); n];
                x_gpu.copy_to_host(&mut x_col)?;
                for i in 0..n {
                    *x_out.get_mut(i, k).unwrap() = x_col[i];
                }
            }

            return Ok(Some(x_out));
        }

        #[cfg(not(feature = "cuda"))]
        Ok(None)
    }
}
