//! Tensor contraction implementation.

use crate::core::scalar::Scalar;
use crate::core::tensor::Tensor;

/// Represents a contraction of two tensors along specified dimensions.
pub struct TensorContraction<'a, T: Scalar, const RANK1: usize, const RANK2: usize> {
    lhs: &'a Tensor<T, RANK1>,
    rhs: &'a Tensor<T, RANK2>,
    lhs_dims: [usize; 2], // Dimensions to contract (simplified for now: 1 pair)
    rhs_dims: [usize; 2],
}

impl<'a, T: Scalar, const RANK1: usize, const RANK2: usize> TensorContraction<'a, T, RANK1, RANK2> {
    pub fn new(
        lhs: &'a Tensor<T, RANK1>,
        rhs: &'a Tensor<T, RANK2>,
        lhs_pair: [usize; 2],
        rhs_pair: [usize; 2],
    ) -> Self {
        Self {
            lhs,
            rhs,
            lhs_dims: lhs_pair,
            rhs_dims: rhs_pair,
        }
    }

    /// Evaluates the contraction.
    /// This is a naive implementation O(N^k).
    /// Real implementation would reshape to matrix multiplication (GEMM).
    /// For this phase, we focus on correctness of the API and logic.
    ///
    /// Result rank = RANK1 + RANK2 - 2 (simplest case: single pair contraction).
    /// Actually, let's implement a specific case: contract one index from each.
    /// Result Rank = RANK1 + RANK2 - 2.
    // Making this generic for any rank return is hard in Rust const generics without `generic_const_exprs`.
    // We will limit to specifically contracting 1 pair of indices for now.
    pub fn eval<const OUT_RANK: usize>(&self) -> Tensor<T, OUT_RANK> {
        // Validation: Contraction dimensions must match size
        let l_dim_idx = self.lhs_dims[0];
        let r_dim_idx = self.rhs_dims[0];

        let l_size = self.lhs.dims()[l_dim_idx];
        let r_size = self.rhs.dims()[r_dim_idx];

        assert_eq!(l_size, r_size, "Contraction dimension sizes must match");

        // Calculate output dimensions
        let mut out_dims = [0; OUT_RANK];
        let mut out_idx = 0;

        for i in 0..RANK1 {
            if i != l_dim_idx {
                out_dims[out_idx] = self.lhs.dims()[i];
                out_idx += 1;
            }
        }
        for i in 0..RANK2 {
            if i != r_dim_idx {
                out_dims[out_idx] = self.rhs.dims()[i];
                out_idx += 1;
            }
        }

        let mut res = Tensor::<T, OUT_RANK>::new(out_dims).unwrap();

        // Naive nested loop evaluation?
        // With generic ranks, we can't write static loops.
        // We iterate over the output tensor, reconstruct indices for lhs/rhs, and sum over the contraction dim.

        let out_size = res.size();
        let k_size = l_size; // The size of the dimension being contracted

        for i in 0..out_size {
            let mut out_indices = [0; OUT_RANK];
            let mut temp = i;
            // Column-major or Row-major? Tensor usually defaults to one. Let's assume standard modular arithmetic.
            for d in 0..OUT_RANK {
                out_indices[d] = temp % out_dims[d];
                temp /= out_dims[d];
            }

            let mut sum = T::default();

            for k in 0..k_size {
                // Construct LHS indices
                let mut l_indices = [0; RANK1];
                let mut out_tracker = 0;
                for d in 0..RANK1 {
                    if d == l_dim_idx {
                        l_indices[d] = k;
                    } else {
                        l_indices[d] = out_indices[out_tracker];
                        out_tracker += 1;
                    }
                }

                // Construct RHS indices
                let mut r_indices = [0; RANK2];
                for d in 0..RANK2 {
                    if d == r_dim_idx {
                        r_indices[d] = k;
                    } else {
                        r_indices[d] = out_indices[out_tracker];
                        out_tracker += 1;
                    }
                }

                let val_l = *self.lhs.get(l_indices).unwrap();
                let val_r = *self.rhs.get(r_indices).unwrap();
                sum += val_l * val_r;
            }

            *res.get_mut(out_indices).unwrap() = sum;
        }

        res
    }
}
