use crate::core::complex::Complex;
use crate::core::scalar::Scalar;
use alloc::vec::Vec;
use rustfft::{num_complex::Complex as NumComplex, FftPlanner};

/// A wrapper for FFT operations using `rustfft`.
pub struct FFT<T: Scalar + rustfft::FftNum> {
    planner: FftPlanner<T>,
}

impl<T: Scalar + rustfft::FftNum> Default for FFT<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + rustfft::FftNum> FFT<T> {
    /// Creates a new FFT planner.
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    /// Computes the forward FFT of the input vector.
    pub fn forward(&mut self, input: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = input.len();
        let fft = self.planner.plan_fft_forward(n);

        let mut buffer: Vec<NumComplex<T>> = input
            .iter()
            .map(|c| NumComplex { re: c.re, im: c.im })
            .collect();

        fft.process(&mut buffer);

        buffer
            .into_iter()
            .map(|c| Complex { re: c.re, im: c.im })
            .collect()
    }

    /// Computes the inverse FFT of the input vector.
    pub fn inverse(&mut self, input: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = input.len();
        let fft = self.planner.plan_fft_inverse(n);

        let mut buffer: Vec<NumComplex<T>> = input
            .iter()
            .map(|c| NumComplex { re: c.re, im: c.im })
            .collect();

        fft.process(&mut buffer);

        buffer
            .into_iter()
            .map(|c| Complex { re: c.re, im: c.im })
            .collect()
    }

    /// Computes the inverse FFT and scales the result by 1/N.
    pub fn inverse_scaled(&mut self, input: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = input.len();
        let mut result = self.inverse(input);
        let scale = <T as Scalar>::from_usize(1) / <T as Scalar>::from_usize(n);
        for elem in result.iter_mut() {
            elem.re *= scale;
            elem.im *= scale;
        }
        result
    }
}

use crate::core::matrix::Matrix;
use crate::core::storage::{DynamicStorage, Storage};

/// Extension trait to add FFT capabilities to Matrix/Vector.
pub trait FftExtension<T: Scalar + rustfft::FftNum> {
    /// Computes the forward FFT of the vector/matrix (flattened).
    fn fft(&self) -> Vec<Complex<T>>;

    /// Computes the inverse FFT of the vector/matrix (flattened).
    fn ifft(&self) -> Vec<Complex<T>>;

    /// Computes 2D forward FFT (Row-wise then Column-wise).
    /// Returns a new Matrix.
    fn fft2(&self) -> Matrix<Complex<T>, DynamicStorage<Complex<T>>>;

    /// Computes 2D inverse FFT (Row-wise then Column-wise).
    /// Returns a new Matrix.
    fn ifft2(&self) -> Matrix<Complex<T>, DynamicStorage<Complex<T>>>;
}

impl<T: Scalar + rustfft::FftNum, S: Storage<Complex<T>>> FftExtension<T>
    for Matrix<Complex<T>, S>
{
    fn fft(&self) -> Vec<Complex<T>> {
        let mut fft = FFT::new();
        let data: Vec<Complex<T>> = (0..self.cols())
            .flat_map(|j| (0..self.rows()).map(move |i| *self.get(i, j).unwrap()))
            .collect();
        fft.forward(&data)
    }

    fn ifft(&self) -> Vec<Complex<T>> {
        let mut fft = FFT::new();
        let data: Vec<Complex<T>> = (0..self.cols())
            .flat_map(|j| (0..self.rows()).map(move |i| *self.get(i, j).unwrap()))
            .collect();
        fft.inverse_scaled(&data)
    }

    fn fft2(&self) -> Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        let rows = self.rows();
        let cols = self.cols();
        let mut res =
            Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(rows, cols).unwrap();

        // Copy initial data
        for j in 0..cols {
            for i in 0..rows {
                *res.get_mut(i, j).unwrap() = *self.get(i, j).unwrap();
            }
        }

        let mut fft = FFT::new();

        // 1. FFT along rows
        for i in 0..rows {
            let mut row_data = Vec::with_capacity(cols);
            for j in 0..cols {
                row_data.push(*res.get(i, j).unwrap());
            }
            let transformed_row = fft.forward(&row_data);
            for (j, val) in transformed_row.iter().enumerate() {
                *res.get_mut(i, j).unwrap() = *val;
            }
        }

        // 2. FFT along columns
        for j in 0..cols {
            let mut col_data = Vec::with_capacity(rows);
            for i in 0..rows {
                col_data.push(*res.get(i, j).unwrap());
            }
            let transformed_col = fft.forward(&col_data);
            for (i, val) in transformed_col.iter().enumerate() {
                *res.get_mut(i, j).unwrap() = *val;
            }
        }

        res
    }

    fn ifft2(&self) -> Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        let rows = self.rows();
        let cols = self.cols();
        let mut res =
            Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(rows, cols).unwrap();

        // Copy initial data
        for j in 0..cols {
            for i in 0..rows {
                *res.get_mut(i, j).unwrap() = *self.get(i, j).unwrap();
            }
        }

        let mut fft = FFT::new();

        // 1. IFFT along rows
        for i in 0..rows {
            let mut row_data = Vec::with_capacity(cols);
            for j in 0..cols {
                row_data.push(*res.get(i, j).unwrap());
            }
            // Use inverse (unscaled) first, we will scale at the very end or at each step?
            // Standard definition: 1/NM * Sum ...
            // If we use inverse_scaled for both, we get (1/N)*(1/M) scaling, which is correct for 2D IFFT.
            let transformed_row = fft.inverse_scaled(&row_data);
            for (j, val) in transformed_row.iter().enumerate() {
                *res.get_mut(i, j).unwrap() = *val;
            }
        }

        // 2. IFFT along columns
        for j in 0..cols {
            let mut col_data = Vec::with_capacity(rows);
            for i in 0..rows {
                col_data.push(*res.get(i, j).unwrap());
            }
            let transformed_col = fft.inverse_scaled(&col_data);
            for (i, val) in transformed_col.iter().enumerate() {
                *res.get_mut(i, j).unwrap() = *val;
            }
        }

        res
    }
}
