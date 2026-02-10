use crate::core::storage::Storage;
use std::marker::PhantomData;

/// Map: Storage wrapper for raw pointers.
/// Allows viewing existing memory as a Matrix without ownership.
///
/// # Safety
/// The caller must ensure the pointer is valid for the lifetime of the Map.
#[derive(Debug)]
pub struct MapStorage<'a, T> {
    ptr: *mut T,
    rows: usize,
    cols: usize,
    stride: usize, // Outer stride (distance between columns)
    _phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Send> Send for MapStorage<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MapStorage<'a, T> {}

impl<'a, T> MapStorage<'a, T> {
    /// Creates a new MapStorage from a raw pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for `rows * cols` elements.
    pub unsafe fn new(ptr: *mut T, rows: usize, cols: usize) -> Self {
        Self {
            ptr,
            rows,
            cols,
            stride: rows, // Default column-major stride
            _phantom: PhantomData,
        }
    }

    /// Creates a new MapStorage with custom stride.
    ///
    /// # Safety
    /// `ptr` must be valid for `rows * cols` elements and the memory layout must match `stride`.
    pub unsafe fn new_with_stride(ptr: *mut T, rows: usize, cols: usize, stride: usize) -> Self {
        Self {
            ptr,
            rows,
            cols,
            stride,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Sync> Storage<T> for MapStorage<'a, T> {
    fn data(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.rows * self.cols) } // Valid only if contiguous with default stride
    }

    fn data_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.rows * self.cols) }
        // Valid only if contiguous
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }

    fn get_ptr(&self, row: usize, col: usize) -> *const T {
        unsafe { self.ptr.add(col * self.stride + row) }
    }
}

// Special case: Clone for MapStorage implies shallow copy of the pointer wrapper, NOT deep copy of data
// This mimics Eigen's Map behavior.
impl<'a, T> Clone for MapStorage<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            rows: self.rows,
            cols: self.cols,
            stride: self.stride,
            _phantom: PhantomData,
        }
    }
}
