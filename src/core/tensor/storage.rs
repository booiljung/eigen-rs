//! Storage and indexing for Tensors.

use crate::core::scalar::Scalar;
use crate::core::storage::{AlignedStorage};

/// Storage for a Tensor, managing its dimensions and flat memory.
pub struct TensorStorage<T: Scalar, const RANK: usize> {
    data: AlignedStorage<T>,
    dims: [usize; RANK],
    strides: [usize; RANK],
}

impl<T: Scalar, const RANK: usize> TensorStorage<T, RANK> {
    pub fn new(dims: [usize; RANK]) -> Result<Self, String> {
        let mut size = 1;
        for &d in &dims {
            size *= d;
        }

        let mut storage = AlignedStorage::new(size, 32)?;
        // Initialize with default values
        for x in storage.as_mut_slice() {
            *x = T::default();
        }

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
        })
    }

    pub fn dims(&self) -> [usize; RANK] {
        self.dims
    }

    pub fn size(&self) -> usize {
        self.data.as_slice().len()
    }

    pub fn get(&self, indices: [usize; RANK]) -> Option<&T> {
        let index = self.flat_index(indices)?;
        Some(&self.data.as_slice()[index])
    }

    pub fn get_mut(&mut self, indices: [usize; RANK]) -> Option<&mut T> {
        let index = self.flat_index(indices)?;
        Some(&mut self.data.as_mut_slice()[index])
    }

    pub fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    fn flat_index(&self, indices: [usize; RANK]) -> Option<usize> {
        let mut flat = 0;
        for i in 0..RANK {
            if indices[i] >= self.dims[i] {
                return None;
            }
            flat += indices[i] * self.strides[i];
        }
        Some(flat)
    }
}
