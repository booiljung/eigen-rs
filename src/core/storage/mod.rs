//! Core storage definitions for eigen-rs.
//! Handles both stack-allocated (fixed) and heap-allocated (dynamic) storage.

use crate::core::scalar::Scalar;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Constant to represent dynamic size.
pub const DYNAMIC: usize = usize::MAX;

/// Trait representing a storage buffer for a matrix.
pub trait Storage<T>: Sync {
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn get_ptr(&self, row: usize, col: usize) -> *const T;

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    fn as_cuda_storage(&self) -> Option<&crate::core::storage::cuda::CudaStorage<T>>
    where
        T: Scalar,
    {
        None
    }
}

/// Aligned memory storage for dense data (Internal helper).
#[derive(Debug, PartialEq, Eq)]
pub struct AlignedStorage<T> {
    ptr: NonNull<T>,
    size: usize,
    // Layout no longer needed as we use fixed 32-byte alignment allocator
}

unsafe impl<T: Send> Send for AlignedStorage<T> {}
unsafe impl<T: Sync> Sync for AlignedStorage<T> {}

impl<T> AlignedStorage<T> {
    pub fn new(size: usize, _alignment: usize) -> Result<Self, String> {
        // Ignore _alignment arg and force 32-byte via allocator
        // Keeping arg for API compatibility if needed, though we only use 32 internally.
        if size == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                size: 0,
            });
        }

        unsafe {
            let ptr = crate::core::allocator::alloc_aligned::<T>(size);
            Ok(Self { ptr, size })
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[T] {
        if self.size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.size == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
        }
    }
}

impl<T: Copy> Clone for AlignedStorage<T> {
    fn clone(&self) -> Self {
        let mut storage = AlignedStorage::new(self.size, 32).unwrap();
        storage.as_mut_slice().copy_from_slice(self.as_slice());
        storage
    }
}

impl<T> Drop for AlignedStorage<T> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                crate::core::allocator::dealloc_aligned(self.ptr, self.size);
            }
        }
    }
}

/// Fixed-size stack storage.
#[derive(Debug, PartialEq, Eq)]
pub struct FixedStorage<T, const R: usize, const C: usize, const S: usize> {
    data: [T; S],
}

impl<T: Copy, const R: usize, const C: usize, const S: usize> Clone for FixedStorage<T, R, C, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy, const R: usize, const C: usize, const S: usize> Copy for FixedStorage<T, R, C, S> {}

impl<T: Default + Copy, const R: usize, const C: usize, const S: usize> FixedStorage<T, R, C, S> {
    pub fn new(rows: usize, cols: usize) -> Result<Self, String> {
        if rows != R || cols != C || rows * cols != S {
            return Err(format!(
                "Size mismatch for FixedStorage: {}x{} != {}x{}",
                rows, cols, R, C
            ));
        }
        Ok(Self {
            data: [T::default(); S],
        })
    }

    pub fn from_array(data: [T; S]) -> Self {
        Self { data }
    }
}

impl<T: Scalar + 'static, const R: usize, const C: usize, const S: usize> Storage<T>
    for FixedStorage<T, R, C, S>
{
    #[inline]
    fn data(&self) -> &[T] {
        &self.data
    }
    #[inline]
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    #[inline]
    fn rows(&self) -> usize {
        R
    }
    #[inline]
    fn cols(&self) -> usize {
        C
    }
    #[inline]
    fn get_ptr(&self, row: usize, col: usize) -> *const T {
        unsafe { self.data.as_ptr().add(col * R + row) }
    }
}

/// Dynamic-size heap storage with alignment.
#[derive(Debug, PartialEq, Eq)]
pub struct DynamicStorage<T> {
    data: AlignedStorage<T>,
    rows: usize,
    cols: usize,
}

impl<T: Default + Copy> DynamicStorage<T> {
    pub fn new(rows: usize, cols: usize) -> Result<Self, String> {
        let mut storage = AlignedStorage::new(rows * cols, 32)?;
        for x in storage.as_mut_slice() {
            *x = T::default();
        }
        Ok(Self {
            data: storage,
            rows,
            cols,
        })
    }

    pub fn from_vec(rows: usize, cols: usize, vec: Vec<T>) -> Result<Self, String> {
        if vec.len() != rows * cols {
            return Err("Vector size mismatch".to_string());
        }
        let mut storage = AlignedStorage::new(rows * cols, 32)?;
        storage.as_mut_slice().copy_from_slice(&vec);
        Ok(Self {
            data: storage,
            rows,
            cols,
        })
    }
}

impl<T: Copy> Clone for DynamicStorage<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T: Scalar + 'static> Storage<T> for DynamicStorage<T> {
    #[inline]
    fn data(&self) -> &[T] {
        self.data.as_slice()
    }
    #[inline]
    fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn get_ptr(&self, row: usize, col: usize) -> *const T {
        unsafe { self.data.as_ptr().add(col * self.rows + row) }
    }
}
pub mod cuda;
pub mod utils;
pub use cuda::CudaStorage;

pub mod map;
pub use map::MapStorage;
pub mod cublas;
