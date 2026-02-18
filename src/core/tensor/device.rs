//! Device abstraction for Tensor operations.

use crate::core::scalar::Scalar;
use crate::core::storage::AlignedStorage;

pub mod cuda;

/// A device capable of allocating memory and executing tensor operations.
pub trait Device: Clone + Default {
    type Storage<T: Scalar>: DeviceStorage<T>;
    
    fn name(&self) -> &'static str;

    // Optional: Methods for cross-device copy could be here or on Storage.
}

/// Abstract storage managed by a device.
/// 
/// We do NOT require AsRef<[T]> / AsMut<[T]> because GPU memory cannot be
/// accessed as a CPU slice directly.
pub trait DeviceStorage<T: Scalar> {
    fn new(size: usize) -> Result<Self, String> where Self: Sized;
    
    /// Returns a slice if the memory is CPU-accessible.
    /// Returns None if memory is on a discrete device (GPU).
    fn as_slice(&self) -> Option<&[T]>;
    
    /// Returns a mutable slice if the memory is CPU-accessible.
    fn as_mut_slice(&mut self) -> Option<&mut [T]>;
}

/// Standard CPU Device.
#[derive(Clone, Default, Debug)]
pub struct CpuDevice;

impl Device for CpuDevice {
    type Storage<T: Scalar> = CpuStorage<T>;
    
    fn name(&self) -> &'static str {
        "CPU"
    }
}

/// CPU Storage using AlignedStorage.
pub struct CpuStorage<T: Scalar> {
    data: AlignedStorage<T>,
}

impl<T: Scalar> convert::AsRef<[T]> for CpuStorage<T> {
    fn as_ref(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<T: Scalar> convert::AsMut<[T]> for CpuStorage<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

impl<T: Scalar> DeviceStorage<T> for CpuStorage<T> {
    fn new(size: usize) -> Result<Self, String> {
        let mut storage = AlignedStorage::new(size, 32)?;
        // Initialize with default
        for x in storage.as_mut_slice() {
            *x = T::default();
        }
        Ok(Self { data: storage })
    }

    fn as_slice(&self) -> Option<&[T]> {
        Some(self.data.as_slice())
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(self.data.as_mut_slice())
    }
}

use std::convert;
