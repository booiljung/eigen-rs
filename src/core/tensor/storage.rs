//! Storage and indexing for Tensors.

use crate::core::scalar::Scalar;
use crate::core::tensor::device::{CpuDevice, Device, DeviceStorage};

/// Storage for a Tensor, managing its dimensions and flat memory.
pub struct TensorStorage<T: Scalar, const RANK: usize, D: Device = CpuDevice> {
    data: D::Storage<T>,
    dims: [usize; RANK],
    strides: [usize; RANK],
    pub device: D,
}

impl<T: Scalar, const RANK: usize> TensorStorage<T, RANK, CpuDevice> {
    pub fn new(dims: [usize; RANK]) -> Result<Self, String> {
        Self::new_with_device(dims, CpuDevice)
    }
}

impl<T: Scalar, const RANK: usize, D: Device> TensorStorage<T, RANK, D> {
    pub fn new_with_device(dims: [usize; RANK], device: D) -> Result<Self, String> {
        let mut size = 1;
        for &d in &dims {
            size *= d;
        }

        let storage = D::Storage::<T>::new(size)?;

        // Calculate strides (Col-Major by default, to match Eigen's Matrix)
        let mut strides = [0; RANK];
        let mut current_stride = 1;
        for i in 0..RANK {
            strides[i] = current_stride;
            current_stride *= dims[i];
        }

        Ok(Self {
            data: storage,
            dims,
            strides,
            device,
        })
    }

    pub fn dims(&self) -> [usize; RANK] {
        self.dims
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn get(&self, indices: [usize; RANK]) -> Option<&T> {
        let index = self.flat_index(indices)?;
        self.data.as_slice().and_then(|s| s.get(index))
    }

    pub fn data(&self) -> &[T] {
        self.data.as_slice().expect("Storage is not CPU-accessible")
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice().expect("Storage is not CPU-accessible")
    }

    pub fn data_opt(&self) -> Option<&[T]> {
        self.data.as_slice()
    }

    pub fn data_mut_opt(&mut self) -> Option<&mut [T]> {
        self.data.as_mut_slice()
    }

    pub fn inner_storage(&self) -> &D::Storage<T> {
        &self.data
    }
    
    pub fn inner_storage_mut(&mut self) -> &mut D::Storage<T> {
        &mut self.data
    }

    fn flat_index(&self, indices: [usize; RANK]) -> Option<usize> {
        let mut flat = 0;
        for (i, idx) in indices.iter().enumerate().take(RANK) {
            if *idx >= self.dims[i] {
                return None;
            }
            flat += *idx * self.strides[i];
        }
        Some(flat)
    }
}
