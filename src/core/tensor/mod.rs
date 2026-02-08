//! Tensor module for multi-dimensional arrays.

use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

pub mod storage;
pub mod ops;
pub mod xpr;
pub mod contraction;
pub mod broadcasting;

pub use storage::TensorStorage;
pub use xpr::TensorXpr;

/// A multi-dimensional tensor.
pub struct Tensor<T: Scalar, const RANK: usize> {
    storage: storage::TensorStorage<T, RANK>,
}

impl<T: Scalar, const RANK: usize> Tensor<T, RANK> {
    pub fn new(dims: [usize; RANK]) -> Result<Self, String> {
        Ok(Self {
            storage: storage::TensorStorage::new(dims)?,
        })
    }

    pub fn dims(&self) -> [usize; RANK] {
        self.storage.dims()
    }

    pub fn size(&self) -> usize {
        self.storage.size()
    }

    pub fn get(&self, indices: [usize; RANK]) -> Option<&T> {
        self.storage.get(indices)
    }

    pub fn get_mut(&mut self, indices: [usize; RANK]) -> Option<&mut T> {
        self.storage.get_mut(indices)
    }

    pub fn data(&self) -> &[T] {
        self.storage.data()
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.storage.data_mut()
    }

    /// Evaluates an expression and assigns it to this tensor.
    pub fn assign<X: xpr::TensorXpr<T, RANK>>(&mut self, xpr: &X) -> Result<(), String> {
        if self.dims() != xpr.dims() {
            return Err("Dimension mismatch in Tensor assignment".to_string());
        }

        let dims = self.dims();
        let size = self.size();
        
        for i in 0..size {
            let mut indices = [0; RANK];
            let mut temp = i;
            // Column-major index reconstruction
            for j in 0..RANK {
                indices[j] = temp % dims[j];
                temp /= dims[j];
            }
            self.data_mut()[i] = xpr.eval(indices);
        }
        Ok(())
    }
    pub fn contract<const RANK2: usize, const OUT_RANK: usize>(
        &self,
        rhs: &Tensor<T, RANK2>,
        lhs_dim: usize,
        rhs_dim: usize
    ) -> Tensor<T, OUT_RANK> {
        let contraction = contraction::TensorContraction::new(self, rhs, [lhs_dim, 0], [rhs_dim, 0]);
        contraction.eval()
    }

    pub fn broadcast(&self, target_dims: [usize; RANK]) -> Result<Tensor<T, RANK>, String> {
        // Materialize broadcasted tensor
        // In a real high-performance library, this would be a virtual view.
        // Here we materialize it for simplicity and correctness verification.
        let broadcast_view = broadcasting::BroadcastedTensor::new(self, target_dims)?;
        let mut res = Tensor::<T, RANK>::new(target_dims)?;
        
        let size = res.size();
        for i in 0..size {
            let mut indices = [0; RANK];
            let mut temp = i;
            for d in 0..RANK {
                indices[d] = temp % target_dims[d];
                temp /= target_dims[d];
            }
            if let Some(val) = broadcast_view.get(indices) {
                *res.get_mut(indices).unwrap() = *val;
            }
        }
        Ok(res)
    }
}

impl<T: Scalar> Tensor<T, 2> {
    pub fn to_matrix(&self) -> Result<crate::core::matrix::MatrixX<T>, String> {
        let dims = self.dims();
        let mut mat = crate::core::matrix::MatrixX::<T>::new_dynamic(dims[0], dims[1])?;
        mat.storage_mut().data_mut().copy_from_slice(self.data());
        Ok(mat)
    }

    pub fn from_matrix(mat: crate::core::matrix::MatrixX<T>) -> Self {
        let mut t = Self::new([mat.rows(), mat.cols()]).unwrap();
        t.data_mut().copy_from_slice(mat.storage().data());
        t
    }
}
