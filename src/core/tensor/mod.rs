//! Tensor module for multi-dimensional arrays.
//!
//! This module provides an N-dimensional generalization of matrices, serving as the foundational
//! infrastructure for Neural Network operations and advanced multi-way data analysis. Key features include:
//! - **N-Dimensional Expressions**: Lazy evaluation of tensor reshaping, broadcasting, and slice operations.
//! - **Tensor Contractions**: High-performance multi-dimensional dot products bridging to GEMM.
//! - **GPU Acceleration**: Built on `CudaDevice` and `CudaStorage` to offload dense element-wise assignments
//!   and contractions directly to the GPU via custom PTX kernels, vastly accelerating Deep Learning workflows.

use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

pub mod broadcasting;
pub mod contraction;
pub mod device;
pub mod ops;
pub mod storage;
pub mod xpr;
pub mod nn;

pub use storage::TensorStorage;
pub use xpr::TensorXpr;

use crate::core::tensor::device::{CpuDevice, Device, DeviceStorage};

/// A multi-dimensional tensor.
pub struct Tensor<T: Scalar, const RANK: usize, D: Device = CpuDevice> {
    storage: storage::TensorStorage<T, RANK, D>,
}

impl<T: Scalar, const RANK: usize, D: Device> Tensor<T, RANK, D> {
    pub fn new_with_device(dims: [usize; RANK], device: D) -> Result<Self, String> {
        Ok(Self {
            storage: storage::TensorStorage::new_with_device(dims, device)?,
        })
    }

    pub fn device(&self) -> &D {
        &self.storage.device
    }

    pub fn dims(&self) -> [usize; RANK] {
        self.storage.dims()
    }

    pub fn size(&self) -> usize {
        self.storage.size()
    }

    pub fn get(&self, indices: [usize; RANK]) -> Option<&T> {
        // Only valid if storage is CPU-accessible
        self.storage.data_opt().and_then(|data| {
            let mut idx = 0;
            let mut stride = 1;
            for i in 0..RANK {
                idx += indices[i] * stride;
                stride *= self.storage.dims()[i];
            }
            data.get(idx)
        })
    }

    // Helper to calculate linear index (ColMajor)
    fn linear_index(&self, indices: [usize; RANK]) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        for i in 0..RANK {
            idx += indices[i] * stride;
            stride *= self.storage.dims()[i];
        }
        idx
    }

    pub fn get_mut(&mut self, indices: [usize; RANK]) -> Option<&mut T> {
        let idx = self.linear_index(indices);
        self.storage
            .data_mut_opt()
            .and_then(|data| data.get_mut(idx))
    }

    pub fn data(&self) -> Option<&[T]> {
        self.storage.data_opt()
    }

    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.storage.data_mut_opt()
    }

    /// Evaluates an expression and assigns it to this tensor.
    pub fn assign<X: xpr::TensorXpr<T, RANK>>(&mut self, xpr: &X) -> Result<(), String> {
        if self.dims() != xpr.dims() {
            return Err("Dimension mismatch in Tensor assignment".to_string());
        }

        let dims = self.dims();
        let size = self.size();

        // Optimized for CPU:
        if let Some(data) = self.data_mut() {
            for i in 0..size {
                let mut indices = [0; RANK];
                let mut temp = i;
                // Column-major index reconstruction
                for j in 0..RANK {
                    indices[j] = temp % dims[j];
                    temp /= dims[j];
                }
                data[i] = xpr.eval(indices);
            }
            Ok(())
        } else {
            #[cfg(feature = "cuda")]
            {
                // Try GPU execution
                if let (Some(device), Some(dest_storage)) =
                    (self.device_as_cuda(), self.as_cuda_storage_mut())
                {
                    if xpr.eval_on_cuda(&device, dest_storage).is_ok() {
                        return Ok(());
                    }
                }
            }

            // Fallback or failure
            Err("Cannot assign generic TensorXpr to GPU Tensor directly (Kernel generation needed or expression not supported on GPU)".to_string())
        }
    }

    // ... (contract methods remain similar, but need to handle data access) ...
    // Note: TensorContraction currently assumes CPU access. It will panic for GPU tensors if not updated.

    pub fn contract<const RANK2: usize, const OUT_RANK: usize>(
        &self,
        rhs: &Tensor<T, RANK2, D>,
        lhs_dim: usize,
        rhs_dim: usize,
    ) -> Tensor<T, OUT_RANK, D> {
        #[cfg(feature = "cuda")]
        if let Some(device) = self.device_as_cuda() {
            // Specialized GPU MatMul path (Rank 2 * Rank 2 -> Rank 2)
            // Contracting dim 1 of LHS with dim 0 of RHS.
            if RANK == 2 && RANK2 == 2 && lhs_dim == 1 && rhs_dim == 0 {
                if OUT_RANK != 2 {
                    panic!("GPU MatMul result must be Rank 2");
                }

                let m = self.dims()[0];
                let k = self.dims()[1];
                let k2 = rhs.dims()[0];
                let n = rhs.dims()[1];

                assert_eq!(k, k2, "Contraction dimension mismatch for GPU MatMul");

                // Validated OUT_RANK=2. Construct dims array.
                // We use unsafe transmute because compiler can't verify const generic equality.
                let out_dims: [usize; OUT_RANK] = unsafe {
                    let d = [m, n];
                    std::ptr::read(&d as *const [usize; 2] as *const [usize; OUT_RANK])
                };
                let device_clone = unsafe { std::mem::transmute_copy(&device) }; // CudaDevice is Clone

                let mut res = Tensor::<T, OUT_RANK, D>::new_with_device(out_dims, device_clone)
                    .expect("Failed to allocate result tensor");

                // Perform MatMul
                // We need &CudaStorage references.
                let lhs_storage = self.as_cuda_storage().expect("LHS must be GPU");
                let rhs_storage = rhs.as_cuda_storage().expect("RHS must be GPU");
                let res_storage = res.as_cuda_storage_mut().expect("Result must be GPU");

                // Call MatMul
                // CudaDevice::matmul(c, a, b, m, n, k)
                // Note: matmul in cuda.rs arguments order: c, a, b.
                device
                    .matmul(res_storage, lhs_storage, rhs_storage, m, n, k)
                    .unwrap();

                return res;
            } else {
                panic!("GPU Contraction only supports standard MatMul (Rank 2, contract 1 with 0) currently.");
            }
        }

        // GPU Contraction requires CuBLAS or Custom Kernel.
        // For now, we only support CPU contraction or panic.
        if self.data().is_none() {
            panic!("GPU Tensor contraction not yet implemented for this case");
        }

        // Convert single dimensions to vectors
        let lhs_pair = vec![lhs_dim];
        let rhs_pair = vec![rhs_dim];

        let contraction = contraction::TensorContraction::new(self, rhs, lhs_pair, rhs_pair);
        contraction.eval()
    }

    pub fn contract_dims<const RANK2: usize, const OUT_RANK: usize>(
        &self,
        rhs: &Tensor<T, RANK2, D>,
        dims: &[(usize, usize)],
    ) -> Tensor<T, OUT_RANK, D> {
        if self.data().is_none() {
            panic!("GPU Tensor contraction not yet implemented");
        }

        let mut lhs_pair = Vec::new();
        let mut rhs_pair = Vec::new();
        for &(l, r) in dims {
            lhs_pair.push(l);
            rhs_pair.push(r);
        }

        let contraction = contraction::TensorContraction::new(self, rhs, lhs_pair, rhs_pair);
        contraction.eval()
    }

    pub fn broadcast(&self, target_dims: [usize; RANK]) -> Result<Tensor<T, RANK, D>, String> {
        // Materialize broadcasted tensor
        // Only CPU support
        let _ = self.data().ok_or("GPU Broadcast not supported")?;

        let broadcast_view = broadcasting::BroadcastedTensor::new(self, target_dims)?;
        let device = self.storage.device.clone();
        let mut res = Tensor::<T, RANK, D>::new_with_device(target_dims, device)?;

        let size = res.size();
        // Since both are on same device (CPU verified), we can unsafe unwrap res.data_mut
        let dest_data = res.data_mut().unwrap();

        for i in 0..size {
            let mut indices = [0; RANK];
            let mut temp = i;
            for d in 0..RANK {
                indices[d] = temp % target_dims[d];
                temp /= target_dims[d];
            }
            if let Some(val) = broadcast_view.get(indices) {
                dest_data[i] = *val;
            }
        }
        Ok(res)
    }

    pub fn permute(&self, new_order: [usize; RANK]) -> Result<Tensor<T, RANK, D>, String> {
        // CPU Check
        // let _ = self.data().ok_or("GPU Permute not supported")?;
        // We remove this check to allow GPU path

        // Validate order
        let mut seen = [false; RANK];
        for &idx in &new_order {
            if idx >= RANK || seen[idx] {
                return Err("Invalid permutation indices".to_string());
            }
            seen[idx] = true;
        }

        // Calculate new dims
        let mut new_dims = [0; RANK];
        for (i, &old_idx) in new_order.iter().enumerate() {
            new_dims[i] = self.dims()[old_idx];
        }

        let device = self.storage.device.clone();
        let mut res = Tensor::<T, RANK, D>::new_with_device(new_dims, device)?;

        // Compute strides for destination manually to avoid borrowing res in loop
        let mut dest_strides = [0; RANK];
        let mut stride = 1;
        for i in 0..RANK {
            dest_strides[i] = stride;
            stride *= new_dims[i];
        }

        if let Some(dest_data) = res.data_mut() {
            // CPU Path
            let src_data = self.data().expect("CPU Tensor must have data");
            let size = self.size();
            for i in 0..size {
                let mut indices = [0; RANK];
                let mut temp = i;
                // Reconstruct indices for SOURCE (Column-major standard iteration of source)
                for d in 0..RANK {
                    indices[d] = temp % self.dims()[d];
                    temp /= self.dims()[d];
                }

                // Map to DEST indices and calculate linear index
                let mut dest_linear = 0;
                for (new_pos, &old_pos) in new_order.iter().enumerate() {
                    dest_linear += indices[old_pos] * dest_strides[new_pos];
                }

                dest_data[dest_linear] = src_data[i]; // Optimized get
            }
        } else {
            #[cfg(feature = "cuda")]
            if let (Some(device), Some(src_storage), Some(dest_storage)) = (
                self.device_as_cuda(),
                self.as_cuda_storage(),
                res.as_cuda_storage_mut(),
            ) {
                // GPU Path
                // Need input strides
                let mut src_strides = [0; RANK];
                let mut stride = 1;
                for i in 0..RANK {
                    src_strides[i] = stride;
                    stride *= self.dims()[i];
                }

                // device.permute takes slices.
                device
                    .permute(
                        dest_storage,
                        src_storage,
                        RANK,
                        &new_dims,
                        &dest_strides,
                        &src_strides,
                        &new_order,
                    )
                    .map_err(|e| format!("GPU Permute failed: {}", e))?;

                return Ok(res);
            }

            return Err(
                "GPU Permute not implemented for this device or feature disabled".to_string(),
            );
        }

        Ok(res)
    }

    pub fn reshape<const NEW_RANK: usize>(
        &self,
        new_dims: [usize; NEW_RANK],
    ) -> Result<Tensor<T, NEW_RANK, D>, String> {
        let current_size = self.size();
        let new_size: usize = new_dims.iter().product();

        if current_size != new_size {
            return Err(format!(
                "Reshape size mismatch: {} vs {}",
                current_size, new_size
            ));
        }

        // Only CPU support for naive copy?
        // Actually, reshape is just a logical view change usually.
        // But if we return a new Tensor with same storage type, we need to populate it.
        // For GPU, we could facilitate D2D copy.

        let device = self.storage.device.clone();
        let mut res = Tensor::<T, NEW_RANK, D>::new_with_device(new_dims, device)?;

        if let (Some(src), Some(dest)) = (self.data(), res.data_mut()) {
            dest.copy_from_slice(src);
        } else {
            #[cfg(feature = "cuda")]
            if let (Some(src_storage), Some(dest_storage)) =
                (self.as_cuda_storage(), res.as_cuda_storage_mut())
            {
                dest_storage
                    .copy_device_to_device(src_storage)
                    .map_err(|e| e)?;
                return Ok(res);
            }

            return Err("GPU Reshape not implemented (Needs D2D Copy)".to_string());
        }

        Ok(res)
    }

    #[cfg(feature = "cuda")]
    fn as_cuda_storage_mut(
        &mut self,
    ) -> Option<&mut crate::core::tensor::device::cuda::CudaStorage<T>> {
        use crate::core::tensor::device::cuda::CudaDevice;
        use std::any::TypeId;
        // Requires D: 'static for TypeId
        if TypeId::of::<D>() == TypeId::of::<CudaDevice>() {
            unsafe {
                let storage_ref = self.storage.inner_storage_mut();
                Some(std::mem::transmute(storage_ref))
            }
        } else {
            None
        }
    }

    #[cfg(feature = "cuda")]
    fn device_as_cuda(&self) -> Option<crate::core::tensor::device::cuda::CudaDevice> {
        use crate::core::tensor::device::cuda::CudaDevice;
        use std::any::TypeId;
        if TypeId::of::<D>() == TypeId::of::<CudaDevice>() {
            unsafe {
                let dev_ref: &CudaDevice = std::mem::transmute(&self.storage.device);
                Some(dev_ref.clone())
            }
        } else {
            None
        }
    }
}

// Device Transfer Methods
impl<T: Scalar, const RANK: usize> Tensor<T, RANK, CpuDevice> {
    pub fn new(dims: [usize; RANK]) -> Result<Self, String> {
        Self::new_with_device(dims, CpuDevice)
    }

    #[cfg(feature = "cuda")]
    pub fn to_device(
        &self,
        device: crate::core::tensor::device::cuda::CudaDevice,
    ) -> Result<Tensor<T, RANK, crate::core::tensor::device::cuda::CudaDevice>, String> {
        use crate::core::tensor::device::cuda::CudaStorage;

        let mut dest =
            Tensor::<T, RANK, crate::core::tensor::device::cuda::CudaDevice>::new_with_device(
                self.dims(),
                device,
            )?;

        // Host -> Device Copy
        // We need access to CudaStorage specific methods.
        // dest.storage is TensorStorage<..., CudaDevice>.
        // storage.storage is CudaStorage.
        // But TensorStorage fields are private.
        // We need to expose a way to do this copy.
        // TensorStorage should maybe implement a trait or expose inner?

        // Solution: Extend TensorStorage to allow copying from slice if D supports it?
        // Or specific method for Cuda?

        // Let's look at `storage::TensorStorage`. It's defined in Mod, let's see.
        // Ideally `dest.storage.copy_from_host(self.data().unwrap())`

        // Since we are in `core/tensor/mod.rs`, `storage` module is child.
        // We can access `dest.storage` field (pub(crate)? no it is private in struct).
        // `storage` field in Tensor is private.

        // We need to add `copy_from_host` to Tensor.
        dest.copy_from_host(self.data().unwrap())?;

        Ok(dest)
    }
}

// Cuda Device Specifics
#[cfg(feature = "cuda")]
impl<T: Scalar, const RANK: usize> Tensor<T, RANK, crate::core::tensor::device::cuda::CudaDevice> {
    pub fn to_host(&self) -> Result<Tensor<T, RANK, CpuDevice>, String> {
        let mut host_tensor = Tensor::<T, RANK, CpuDevice>::new(self.dims())?;

        self.copy_to_host(host_tensor.data_mut().unwrap())?;

        Ok(host_tensor)
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), String> {
        self.storage.inner_storage_mut().copy_from_host(src)
    }

    pub fn copy_to_host(&self, dest: &mut [T]) -> Result<(), String> {
        self.storage.inner_storage().copy_to_host(dest)
    }
}

impl<T: Scalar, const RANK: usize, D: Device> Tensor<T, RANK, D> {
    pub fn to_matrix(&self) -> Result<crate::core::matrix::MatrixX<T>, String> {
        if RANK != 2 {
            return Err(format!(
                "to_matrix only supported for Rank 2 tensors, got {}",
                RANK
            ));
        }
        let dims = self.dims();
        let dims_slice = dims.as_slice();
        let rows = dims_slice[0];
        let cols = dims_slice[1];
        let mut mat = crate::core::matrix::MatrixX::new_dynamic(rows, cols)?;

        let dest_data = mat.storage_mut().data_mut();

        if let Some(src_data) = self.data() {
            dest_data.copy_from_slice(src_data);
            Ok(mat)
        } else {
            Err("Cannot convert GPU Tensor to Matrix directly. Use to_host() first.".to_string())
        }
    }

    pub fn from_matrix_reshaped(
        mat: crate::core::matrix::MatrixX<T>,
        device: D,
        new_dims: [usize; RANK],
    ) -> Result<Self, String> {
        let mat_size = mat.rows() * mat.cols();
        let tensor_size: usize = new_dims.iter().product();
        if mat_size != tensor_size {
            return Err(format!(
                "Size mismatch in from_matrix_reshaped: Matrix {} != Tensor {}",
                mat_size, tensor_size
            ));
        }

        let mut t = Self::new_with_device(new_dims, device)?;

        let src_data = mat.storage().data();

        if let Some(dest_data) = t.data_mut() {
            dest_data.copy_from_slice(src_data);
            Ok(t)
        } else {
            // Host -> GPU
            // This generic method doesn't know about CudaStorage specific copy_from_host.
            // We need `DeviceStorage` to expose `copy_from_slice`?
            // But `copy_from_slice` usually implies `&[T] -> &[T]` (memcpy).
            // H2D copy is special.

            // Workaround: We cannot support generic `from_matrix_reshaped` for GPU directly
            // unless we expand `Device` trait to handle H2D copy generic.
            Err("Direct Matrix->GPU Tensor conversion not supported in generic method. Create Host Tensor then to_device().".to_string())
        }
    }
}

// TensorXpr Implementations with CUDA hooks
// Implemented here to allow access to `storage.inner_storage()`
use crate::core::tensor::device::cuda::{CudaDevice, CudaStorage};
use std::any::TypeId;

impl<T: Scalar, const RANK: usize, D: Device + 'static> TensorXpr<T, RANK> for Tensor<T, RANK, D> {
    fn dims(&self) -> [usize; RANK] {
        self.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        *self.get(indices).unwrap_or(&T::default())
    }

    #[cfg(feature = "cuda")]
    fn as_cuda_storage(&self) -> Option<&CudaStorage<T>> {
        if TypeId::of::<D>() == TypeId::of::<CudaDevice>() {
            // Safety: Checked TypeId matches CudaDevice.
            // TensorStorage<..., CudaDevice> contains CudaStorage<T>.
            // inner_storage() returns &D::Storage<T>, which is &CudaStorage<T>.
            unsafe {
                let storage_ref = self.storage.inner_storage();
                Some(std::mem::transmute(storage_ref))
            }
        } else {
            None
        }
    }

    #[cfg(feature = "cuda")]
    fn eval_on_cuda(&self, device: &CudaDevice, out: &mut CudaStorage<T>) -> Result<(), ()> {
        if let Some(src) = self.as_cuda_storage() {
            device.assign(out, src).map_err(|_| ())
        } else {
            Err(())
        }
    }
}

impl<T: Scalar, const RANK: usize, D: Device + 'static> TensorXpr<T, RANK> for &Tensor<T, RANK, D> {
    fn dims(&self) -> [usize; RANK] {
        (*self).dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        *self.get(indices).unwrap_or(&T::default())
    }

    #[cfg(feature = "cuda")]
    fn as_cuda_storage(&self) -> Option<&CudaStorage<T>> {
        (*self).as_cuda_storage()
    }

    #[cfg(feature = "cuda")]
    fn eval_on_cuda(&self, device: &CudaDevice, out: &mut CudaStorage<T>) -> Result<(), ()> {
        (*self).eval_on_cuda(device, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::device::CpuDevice;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::<f32, 2>::new([2, 3]).unwrap();
        assert_eq!(t.dims(), [2, 3]);
        assert_eq!(t.size(), 6);
    }

    #[test]
    fn test_tensor_access() {
        let mut t = Tensor::<f32, 2>::new([2, 2]).unwrap();
        *t.get_mut([0, 0]).unwrap() = 1.0;
        *t.get_mut([1, 1]).unwrap() = 2.0;

        assert_eq!(t.get([0, 0]), Some(&1.0));
        assert_eq!(t.get([1, 1]), Some(&2.0));
        assert_eq!(t.get([0, 1]), Some(&0.0)); // Default initialization
    }

    #[test]
    fn test_tensor_contraction() {
        // 2x2 Identity * 2x2 Identity
        let mut t1 = Tensor::<f32, 2>::new([2, 2]).unwrap();
        *t1.get_mut([0, 0]).unwrap() = 1.0;
        *t1.get_mut([1, 1]).unwrap() = 1.0;

        let mut t2 = Tensor::<f32, 2>::new([2, 2]).unwrap();
        *t2.get_mut([0, 0]).unwrap() = 1.0;
        *t2.get_mut([1, 1]).unwrap() = 1.0;

        // Contract dim 1 of t1 with dim 0 of t2
        let t3 = t1.contract(&t2, 1, 0);

        assert_eq!(t3.dims(), [2, 2]);
        assert_eq!(t3.get([0, 0]), Some(&1.0));
        assert_eq!(t3.get([1, 1]), Some(&1.0));
        assert_eq!(t3.get([1, 0]), Some(&0.0));
    }

    #[test]
    fn test_tensor_multi_dim_contraction() {
        // 2x2x2
        let mut t1 = Tensor::<f32, 3>::new([2, 2, 2]).unwrap();
        for x in t1.data_mut().unwrap() {
            *x = 1.0;
        }

        let mut t2 = Tensor::<f32, 3>::new([2, 2, 2]).unwrap();
        for x in t2.data_mut().unwrap() {
            *x = 2.0;
        }

        // Contract (1, 0) and (2, 1)
        let dims = [(1, 0), (2, 1)];
        let t3: Tensor<f32, 2> = t1.contract_dims(&t2, &dims);

        assert_eq!(t3.dims(), [2, 2]);
        for x in t3.data().unwrap() {
            assert_eq!(*x, 8.0);
        }
    }

    #[test]
    fn test_tensor_broadcasting() {
        // 2x1 Tensor
        let mut t = Tensor::<f32, 2>::new([2, 1]).unwrap();
        *t.get_mut([0, 0]).unwrap() = 10.0;
        *t.get_mut([1, 0]).unwrap() = 20.0;

        // Broadcast to 2x2
        let t_broad = t.broadcast([2, 2]).unwrap();

        assert_eq!(t_broad.dims(), [2, 2]);
        // Col 0
        assert_eq!(t_broad.get([0, 0]), Some(&10.0));
        assert_eq!(t_broad.get([1, 0]), Some(&20.0));
        // Col 1 (replicated)
        assert_eq!(t_broad.get([0, 1]), Some(&10.0));
        assert_eq!(t_broad.get([1, 1]), Some(&20.0));
    }
}
