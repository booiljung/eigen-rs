use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// 32-byte alignment for AVX2
pub const ALIGNMENT: usize = 32;

/// Allocate memory with 32-byte alignment.
/// Panics if layout creation fails or allocation fails.
pub unsafe fn alloc_aligned<T>(capacity: usize) -> NonNull<T> {
    if capacity == 0 {
        return NonNull::dangling();
    }

    let layout = Layout::from_size_align(
        capacity * std::mem::size_of::<T>(),
        ALIGNMENT,
    ).expect("Failed to create layout");

    let ptr = alloc(layout) as *mut T;
    NonNull::new(ptr).expect("Failed to allocate memory")
}

/// Deallocate memory with 32-byte alignment.
pub unsafe fn dealloc_aligned<T>(ptr: NonNull<T>, capacity: usize) {
    if capacity == 0 {
        return;
    }

    let layout = Layout::from_size_align(
        capacity * std::mem::size_of::<T>(),
        ALIGNMENT,
    ).expect("Failed to create layout");

    dealloc(ptr.as_ptr() as *mut u8, layout);
}
