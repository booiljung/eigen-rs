//! Operator implementations for sparse matrices.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::storage::{DynamicStorage, Storage};
use std::ops::{Add, Mul, Sub};

impl<T: Scalar> Add for &SparseMatrix<T> {
    type Output = Result<SparseMatrix<T>, String>;
    fn add(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |a, b| a + b)
    }
}

impl<T: Scalar> Sub for &SparseMatrix<T> {
    type Output = Result<SparseMatrix<T>, String>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |a, b| a - b)
    }
}

impl<T: Scalar> Mul<T> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut res = self.clone();
        res.scale(rhs);
        res
    }
}

impl<T: Scalar> Mul for &SparseMatrix<T> {
    type Output = Result<SparseMatrix<T>, String>;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols() != rhs.rows() {
            return Err("Incompatible dimensions for sparse multiplication".to_string());
        }
        if self.order() != rhs.order() {
            return Err("Sparse multiplication requires same storage order".to_string());
        }

        let outer_dim_rhs = rhs.cols();
        let outer_dim_lhs = self.rows();

        let mut res_values = Vec::new();
        let mut res_inner = Vec::new();

        match self.order() {
            StorageOrder::RowMajor => {
                let mut res_outer = vec![0; outer_dim_lhs + 1];
                let mut workspace = vec![T::default(); outer_dim_rhs];
                let mut marker = vec![false; outer_dim_rhs];
                let mut active_cols = Vec::with_capacity(outer_dim_rhs);

                for i in 0..outer_dim_lhs {
                    let mut it_a = InnerIterator::new(self, i);
                    while it_a.is_valid() {
                        let val_a = it_a.value();
                        let k = it_a.index();

                        let mut it_b = InnerIterator::new(rhs, k);
                        while it_b.is_valid() {
                            let val_b = it_b.value();
                            let j = it_b.index();

                            workspace[j] += val_a * val_b;
                            if !marker[j] {
                                marker[j] = true;
                                active_cols.push(j);
                            }
                            it_b.next();
                        }
                        it_a.next();
                    }

                    active_cols.sort_unstable();
                    for &j in &active_cols {
                        let val = workspace[j];
                        if val != T::default() {
                            res_values.push(val);
                            res_inner.push(j);
                        }
                        workspace[j] = T::default();
                        marker[j] = false;
                    }
                    active_cols.clear();
                    res_outer[i + 1] = res_values.len();
                }
                Ok(SparseMatrix::from_raw(
                    self.rows(),
                    rhs.cols(),
                    res_values,
                    res_inner,
                    res_outer,
                    StorageOrder::RowMajor,
                ))
            }
            StorageOrder::ColMajor => {
                let mut res_outer = vec![0; outer_dim_rhs + 1];
                let mut workspace = vec![T::default(); outer_dim_lhs];
                let mut marker = vec![false; outer_dim_lhs];
                let mut active_rows = Vec::with_capacity(outer_dim_lhs);

                for j in 0..outer_dim_rhs {
                    let mut it_b = InnerIterator::new(rhs, j);
                    while it_b.is_valid() {
                        let val_b = it_b.value();
                        let k = it_b.index();

                        let mut it_a = InnerIterator::new(self, k);
                        while it_a.is_valid() {
                            let val_a = it_a.value();
                            let i = it_a.index();

                            workspace[i] += val_a * val_b;
                            if !marker[i] {
                                marker[i] = true;
                                active_rows.push(i);
                            }
                            it_a.next();
                        }
                        it_b.next();
                    }

                    active_rows.sort_unstable();
                    for &i in &active_rows {
                        let val = workspace[i];
                        if val != T::default() {
                            res_values.push(val);
                            res_inner.push(i);
                        }
                        workspace[i] = T::default();
                        marker[i] = false;
                    }
                    active_rows.clear();
                    res_outer[j + 1] = res_values.len();
                }
                Ok(SparseMatrix::from_raw(
                    self.rows(),
                    rhs.cols(),
                    res_values,
                    res_inner,
                    res_outer,
                    StorageOrder::ColMajor,
                ))
            }
        }
    }
}

impl<T: Scalar, S: Storage<T>> Mul<&Matrix<T, S>> for &SparseMatrix<T> {
    type Output = Result<Matrix<T, DynamicStorage<T>>, String>;
    fn mul(self, rhs: &Matrix<T, S>) -> Self::Output {
        self.mul_dense(rhs)
    }
}

impl<T: Scalar> SparseMatrix<T> {
    fn binary_op<F>(&self, rhs: &Self, op: F) -> Result<Self, String>
    where
        F: Fn(T, T) -> T,
    {
        if self.rows() != rhs.rows() || self.cols() != rhs.cols() {
            return Err("Incompatible dimensions for sparse operation".to_string());
        }
        if self.order() != rhs.order() {
            return Err("Sparse operation requires same storage order".to_string());
        }

        let outer_limit = if self.order() == StorageOrder::RowMajor {
            self.rows()
        } else {
            self.cols()
        };

        let mut res_values = Vec::new();
        let mut res_inner = Vec::new();
        let mut res_outer = vec![0; outer_limit + 1];

        for i in 0..outer_limit {
            let mut it_a = InnerIterator::new(self, i);
            let mut it_b = InnerIterator::new(rhs, i);

            while it_a.is_valid() || it_b.is_valid() {
                if it_a.is_valid() && it_b.is_valid() {
                    let idx_a = it_a.index();
                    let idx_b = it_b.index();
                    if idx_a == idx_b {
                        let val = op(it_a.value(), it_b.value());
                        if val != T::default() {
                            res_values.push(val);
                            res_inner.push(idx_a);
                        }
                        it_a.next();
                        it_b.next();
                    } else if idx_a < idx_b {
                        res_values.push(it_a.value());
                        res_inner.push(idx_a);
                        it_a.next();
                    } else {
                        res_values.push(op(T::default(), it_b.value()));
                        res_inner.push(idx_b);
                        it_b.next();
                    }
                } else if it_a.is_valid() {
                    res_values.push(it_a.value());
                    res_inner.push(it_a.index());
                    it_a.next();
                } else {
                    res_values.push(op(T::default(), it_b.value()));
                    res_inner.push(it_b.index());
                    it_b.next();
                }
            }
            res_outer[i + 1] = res_values.len();
        }

        Ok(SparseMatrix::from_raw(
            self.rows(),
            self.cols(),
            res_values,
            res_inner,
            res_outer,
            self.order(),
        ))
    }
}
