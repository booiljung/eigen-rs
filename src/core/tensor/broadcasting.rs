//! Tensor broadcasting implementation.

use crate::core::scalar::Scalar;
use crate::core::tensor::Tensor;

/// Represents a broadcasted tensor.
/// This is a virtual view that repeats dimensions of size 1 to match a target shape.
pub struct BroadcastedTensor<'a, T: Scalar, const RANK: usize> {
    tensor: &'a Tensor<T, RANK>,
    target_dims: [usize; RANK],
}

impl<'a, T: Scalar, const RANK: usize> BroadcastedTensor<'a, T, RANK> {
    pub fn new(tensor: &'a Tensor<T, RANK>, target_dims: [usize; RANK]) -> Result<Self, String> {
        let src_dims = tensor.dims();
        for i in 0..RANK {
            if src_dims[i] != target_dims[i] && src_dims[i] != 1 {
                return Err(format!(
                    "Cannot broadcast dimension {} of size {} to {}",
                    i, src_dims[i], target_dims[i]
                ));
            }
        }
        Ok(Self {
            tensor,
            target_dims,
        })
    }

    pub fn get(&self, indices: [usize; RANK]) -> Option<&T> {
        let mut src_indices = [0; RANK];
        let src_dims = self.tensor.dims();
        for i in 0..RANK {
            if indices[i] >= self.target_dims[i] {
                return None;
            }
            if src_dims[i] == 1 {
                src_indices[i] = 0;
            } else {
                src_indices[i] = indices[i];
            }
        }
        self.tensor.get(src_indices)
    }
}
