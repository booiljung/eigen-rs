//! Storage utilities for eigen-rs.

/// A Send-safe wrapper for raw pointers.
/// Used for parallel evaluation where each thread accesses distinct regions.
#[derive(Clone, Copy)]
pub struct PtrWrapper<T>(pub *mut T);

unsafe impl<T> Send for PtrWrapper<T> {}
unsafe impl<T> Sync for PtrWrapper<T> {}
