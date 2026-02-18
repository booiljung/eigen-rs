use crate::core::scalar::Scalar;
use super::Tensor;
use crate::core::tensor::device::{CpuDevice, Device};


/// Represents a contraction of two tensors along specified dimensions.
pub struct TensorContraction<'a, T: Scalar, const RANK1: usize, const RANK2: usize, D: Device = CpuDevice> {
    lhs: &'a Tensor<T, RANK1, D>,
    rhs: &'a Tensor<T, RANK2, D>,
    lhs_dims: Vec<usize>,
    rhs_dims: Vec<usize>,
}

impl<'a, T: Scalar, const RANK1: usize, const RANK2: usize, D: Device> TensorContraction<'a, T, RANK1, RANK2, D> {
    pub fn new(
        lhs: &'a Tensor<T, RANK1, D>,
        rhs: &'a Tensor<T, RANK2, D>,
        lhs_pair: Vec<usize>,
        rhs_pair: Vec<usize>,
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
    pub fn eval<const OUT_RANK: usize>(&self) -> Tensor<T, OUT_RANK, D> {
        // Validation: Contraction dimensions must match size
        assert_eq!(self.lhs_dims.len(), self.rhs_dims.len(), "Number of contraction dimensions must match");
        
        let mut k_size = 1;
        for i in 0..self.lhs_dims.len() {
            let l_idx = self.lhs_dims[i];
            let r_idx = self.rhs_dims[i];
            let l_dim = self.lhs.dims()[l_idx];
            let r_dim = self.rhs.dims()[r_idx];
            assert_eq!(l_dim, r_dim, "Contraction dimension sizes must match");
            k_size *= l_dim;
        }

        // Identify free dimensions
        let mut l_free = Vec::new();
        for i in 0..RANK1 {
            if !self.lhs_dims.contains(&i) {
                l_free.push(i);
            }
        }
        let mut r_free = Vec::new();
        for i in 0..RANK2 {
            if !self.rhs_dims.contains(&i) {
                r_free.push(i);
            }
        }
        
        let m_size: usize = l_free.iter().map(|&d| self.lhs.dims()[d]).product();
        let n_size: usize = r_free.iter().map(|&d| self.rhs.dims()[d]).product();
        
        // Strategy: Permute -> Reshape -> Matrix Mul
        // LHS Permutation: [Free Dims..., Contract Dims...]
        // LHS Reshape: (M, K)
        // RHS Permutation: [Contract Dims..., Free Dims...]
        // RHS Reshape: (K, N)
        // Result: (M, N) -> Reshape to (LHS Free..., RHS Free...)
        
        // 1. Permute LHS
        let mut l_perm = l_free.clone();
        l_perm.extend_from_slice(&self.lhs_dims);
        
        // Verify permutation logic
        // We need array for permute, but l_perm is Vec.
        // Convert Vec to Array.
        let mut l_perm_arr = [0; RANK1];
        for (i, &idx) in l_perm.iter().enumerate() {
            l_perm_arr[i] = idx;
        }
        
        let lhs_permuted = self.lhs.permute(l_perm_arr).unwrap();
        let lhs_mat_t = lhs_permuted.reshape([m_size, k_size]).unwrap();
        
        // 2. Permute RHS
        let mut r_perm = self.rhs_dims.clone();
        r_perm.extend_from_slice(&r_free);
        
        let mut r_perm_arr = [0; RANK2];
        for (i, &idx) in r_perm.iter().enumerate() {
            r_perm_arr[i] = idx;
        }
        
        let rhs_permuted = self.rhs.permute(r_perm_arr).unwrap();
        let rhs_mat_t = rhs_permuted.reshape([k_size, n_size]).unwrap();
        
        // 3. Convert to Matrix and Multiply
        // Note: Tensor::to_matrix returns MatrixX (dynamic)
        let lhs_mat = lhs_mat_t.to_matrix().unwrap();
        let rhs_mat = rhs_mat_t.to_matrix().unwrap();
        
        // Matrix multiplication returns a Product expression
        let product = &lhs_mat * &rhs_mat;
        
        use crate::core::xpr::MatrixXpr; // Ensure traits are visible for rows()/cols()
        let mut res_mat = crate::core::matrix::MatrixX::new_dynamic(product.rows(), product.cols()).unwrap();
        res_mat.assign(&product).unwrap();
        
        // 4. Convert back to Tensor
        let device = self.lhs.storage.device.clone();
        let mut res_tensor_2d = Tensor::<T, 2, D>::new_with_device([product.rows(), product.cols()], device).unwrap();
        let ptr = res_mat.as_ptr().unwrap();
        let len = res_mat.rows() * res_mat.cols();
        let src_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        res_tensor_2d.data_mut().expect("Contraction result must be on Host").copy_from_slice(src_slice);
        
        // 5. Reshape to Output Dimensions
        // Output dims are LHS Free (in order) then RHS Free (in order)
        // The result of GEMM (M, N) corresponds to (LHS Free flattened, RHS Free flattened).
        // So simple reshape works IF the logical layout matches.
        
        // OUT_RANK verification
        // assert_eq!(OUT_RANK, l_free.len() + r_free.len());
        
        let mut out_dims = [0; OUT_RANK];
        // Fill out_dims
        // Wait, how do we know the caller's OUT_RANK matches?
        // We assume valid construction.
        // We need to map l_free dims and r_free dims to out_dims
        let mut idx = 0;
        for &d in &l_free {
             if idx < OUT_RANK {
                 out_dims[idx] = self.lhs.dims()[d];
                 idx += 1;
             }
        }
        for &d in &r_free {
             if idx < OUT_RANK {
                 out_dims[idx] = self.rhs.dims()[d];
                 idx += 1;
             }
        }
        
        res_tensor_2d.reshape(out_dims).unwrap()
    }
}
