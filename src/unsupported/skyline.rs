//! Skyline (Profile/Envelope) Matrix Storage
//!
//! Specialized storage for symmetric matrices with variable bandwidth, common in FEM.
//! Stores the lower triangle profile.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::SparseMatrix;
use crate::core::storage::DynamicStorage;
use alloc::vec;
use alloc::vec::Vec;

/// Indexing scheme:
/// For each row `i`, we store elements `A[i, j]` for `k <= j <= i`,
/// where `k` is the first non-zero column index in that row.
///
/// `diag`: Stores A[i, i]
/// `lower`: Stores A[i, j] for j < i. Encoded consecutively.
/// `profile_ptrs`: `profile_ptrs[i]` is the starting index in `lower` for row `i`.
///                 `profile_ptrs[i+1]` is the end index.
#[derive(Debug, Clone)]
pub struct SkylineStorage<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub diag: Vec<T>,
    pub lower: Vec<T>,
    pub profile_ptrs: Vec<usize>,
}

impl<T: Scalar> SkylineStorage<T> {
    pub fn new(size: usize) -> Self {
        Self {
            rows: size,
            cols: size,
            diag: vec![T::zero(); size],
            lower: Vec::new(),
            profile_ptrs: vec![0; size + 1],
        }
    }
}

pub struct SkylineMatrix<T: Scalar> {
    pub storage: SkylineStorage<T>,
}

impl<T: Scalar> SkylineMatrix<T> {
    pub fn new(size: usize) -> Self {
        Self {
            storage: SkylineStorage::new(size),
        }
    }

    pub fn rows(&self) -> usize {
        self.storage.rows
    }

    pub fn cols(&self) -> usize {
        self.storage.cols
    }

    /// Access reference to element at (row, col).
    /// Currently O(1) if diagonal, O(1) if in profile lookup? No.
    /// Access is tricky: We know the pointer to the row start.
    /// The row stores elements ending at `i-1`.
    /// The length of row `i` profile is `ptr[i+1] - ptr[i]`.
    /// The column index of the last element in `lower` for row `i` is `i-1`.
    /// So `lower[ptr[i+1]-1]` corresponds to `A[i, i-1]`.
    /// `lower[ptr[i]]` corresponds to `A[i, i - len]`.
    /// The column index `j` corresponds to an offset `offset = j - (i - len)`.
    pub fn coeff(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows() && col < self.cols());

        let (r, c) = if row >= col { (row, col) } else { (col, row) };

        if r == c {
            return self.storage.diag[r];
        }

        // Check if (r, c) is within the stored profile
        let start_idx = self.storage.profile_ptrs[r];
        let end_idx = self.storage.profile_ptrs[r + 1];
        let len = end_idx - start_idx;

        // The stored range for row r matches columns: [r - len, r - 1]
        let first_col = r.saturating_sub(len);

        if c >= first_col && c < r {
            let offset = c - first_col;
            // The storage is contiguous for the row.
            return self.storage.lower[start_idx + offset];
        }

        T::zero()
    }

    /// Set a value. NOTE: This cannot change the profile structure (topology).
    /// If you try to set a non-zero outside the profile, it will panic or ignore (we panic for safety).
    pub fn coeff_mut(&mut self, row: usize, col: usize) -> &mut T {
        assert!(row < self.rows() && col < self.cols());
        let (r, c) = if row >= col { (row, col) } else { (col, row) };

        if r == c {
            return &mut self.storage.diag[r];
        }

        let start_idx = self.storage.profile_ptrs[r];
        let end_idx = self.storage.profile_ptrs[r + 1];
        let len = end_idx - start_idx;
        let first_col = r.saturating_sub(len);

        if c >= first_col && c < r {
            let offset = c - first_col;
            return &mut self.storage.lower[start_idx + offset];
        }

        panic!("SkylineMatrix topology is fixed. Cannot set coefficient outside profile.");
    }

    // Constructor from Dense Matrix (allocates envelope)
    pub fn from_dense(mat: &Matrix<T, DynamicStorage<T>>) -> Self {
        assert_eq!(mat.rows(), mat.cols(), "SkylineMatrix must be square");
        let n = mat.rows();
        let mut skyl = Self::new(n);

        // 1. Determine profile
        let mut profile_ptrs = vec![0usize; n + 1];
        let mut current_ptr = 0;

        for (i, ptr) in profile_ptrs.iter_mut().enumerate().take(n) {
            *ptr = current_ptr;
            // Find first non-zero column k < i
            let mut first_nz = i;
            for k in 0..i {
                // Ensure symmetric check? Assuming symmetric input for structure, OR check both if generic.
                // Standard: check (i, k) since we store lower part.
                if !mat.get(i, k).unwrap().is_zero() {
                    first_nz = k;
                    break;
                }
            }
            if first_nz < i {
                current_ptr += i - first_nz;
            }
        }
        profile_ptrs[n] = current_ptr;

        // 2. Allocate
        skyl.storage.profile_ptrs = profile_ptrs;
        skyl.storage.lower = vec![T::zero(); current_ptr];

        // 3. Fill
        for i in 0..n {
            skyl.storage.diag[i] = *mat.get(i, i).unwrap();

            let start = skyl.storage.profile_ptrs[i];
            let end = skyl.storage.profile_ptrs[i + 1];
            let len = end - start;
            if len > 0 {
                let first_col = i - len;
                for k in 0..len {
                    let col = first_col + k;
                    skyl.storage.lower[start + k] = *mat.get(i, col).unwrap();
                }
            }
        }
        skyl
    }

    /// Construct from Sparse Matrix.
    /// Efficiently determines profile from non-zero structure.
    pub fn from_sparse(mat: &SparseMatrix<T>) -> Self {
        assert_eq!(mat.rows(), mat.cols(), "SkylineMatrix must be square");
        let n = mat.rows();
        let mut skyl = Self::new(n);

        // 1. Determine profile
        let mut profile_ptrs = vec![0usize; n + 1];

        // Loop removed (redundant initialization)

        // Better approach:
        // 1. Initialize min_col[i] = i for all i.
        // 2. Iterate over all non-zeros (i, j, v) of mat.
        //    If j < i, min_col[i] = min(min_col[i], j).
        //    (We only care about lower triangle)

        let mut min_col = (0..n).collect::<Vec<_>>();

        // Sparse iterator is needed.
        // `mat.triplet_iter()`? or `iter()`?
        // Let's check SparseMatrix API. It usually has `outer_iterator`.

        for k in 0..mat.outer_size() {
            let mut it = InnerIterator::new(mat, k);
            while it.is_valid() {
                let (curr_row, curr_col) = (it.row(), it.col());
                // If CSR: k is row, curr_col is col.
                // If CSC: k is col, curr_row is row.
                // We need (row, col).
                // Let's rely on standard iteration.
                if curr_col < curr_row {
                    if curr_col < min_col[curr_row] {
                        min_col[curr_row] = curr_col;
                    }
                } else if curr_row < curr_col {
                    // Upper triangle: symmetric assumption?
                    if curr_row < min_col[curr_col] {
                        min_col[curr_col] = curr_row;
                    }
                }
                it.next(); // FIX: Advancement was missing
            }
        }

        // 3. Build profile_ptrs
        let mut current_ptr = 0;
        for i in 0..n {
            profile_ptrs[i] = current_ptr;
            if min_col[i] < i {
                current_ptr += i - min_col[i];
            }
        }
        profile_ptrs[n] = current_ptr;

        // 4. Allocate and Fill
        skyl.storage.profile_ptrs = profile_ptrs;
        skyl.storage.lower = vec![T::zero(); current_ptr];

        // Fill Values
        for k in 0..mat.outer_size() {
            let mut it = InnerIterator::new(mat, k);
            while it.is_valid() {
                let (r, c) = (it.row(), it.col());
                let val = it.value();
                if r == c {
                    skyl.storage.diag[r] = val;
                } else {
                    // We store lower part.
                    // If val is in upper part (r < c), we can ignore it?
                    // Or assumes symmetric input so A[r,c] == A[c,r]?
                    // Skyline typically used given a symmetric matrix.
                    // Let's store A[r,c] into lower slot (c, r) if r < c.
                    // Wait, coefficients might be different if non-symmetric structure but symmetric profile?
                    // Usually for LDLT we only read lower part.
                    // But if user passes full sparse, we might want to sum duplicates or just take lower?
                    // Let's only read lower part elements (r > c) and diagonal.

                    if r > c {
                        let start = skyl.storage.profile_ptrs[r];
                        let end = skyl.storage.profile_ptrs[r + 1];
                        let len = end - start;
                        let first_col = r - len;

                        if c >= first_col {
                            let offset = c - first_col;
                            skyl.storage.lower[start + offset] = val;
                        }
                    }
                    // If symmetric and only upper provided?
                    // Let's assume input must provide lower part for now.
                    // Or we should handle (c, r) if r < c.
                    // Let's handle both for robustness regarding structure,
                    // but for value, last one wins or sum?
                    // Standard Sparse `coeff` usually sums triplets on construction but `inner_iterator` iterates unique stored elements.
                    // Let's stick to reading only strictly lower part + diagonal.
                    // Explicitly: "SkylineMatrix from Sparse assumes input contains the lower triangle values."
                }
                it.next(); // FIX: Added missing next()
            }
        }

        skyl
    }
}
