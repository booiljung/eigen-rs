use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;
#[cfg(feature = "cuda")]
use crate::core::storage::cuda::CudaStorage;

/// Available execution hardware backends.
#[derive(Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Pure scalar execution without packing overheads. Optimal for N < 32.
    Scalar,
    /// Multithreaded AVX2/FMA blocked execution. Optimal for 32 <= N < 512.
    Simd,
    /// cuBLAS execution on the device. Optimal for N >= 512.
    Cuda,
}

/// A runtime heuristic dispatcher selecting the most efficient backend 
/// based on matrix dimensions.
pub fn select_backend(rows: usize, cols: usize, depth: usize) -> Architecture {
    // The maximum dimension dominates the computational intensity (O(N^3))
    let max_dim = rows.max(cols).max(depth);
    
    if max_dim < 32 {
        Architecture::Scalar
    } else if cfg!(feature = "cuda") && max_dim >= 512 {
        Architecture::Cuda
    } else {
        Architecture::Simd
    }
}

/// Demonstrates how the backend is physically dispatched based on the selected heuristic.
/// This acts as the runtime entry point scaling gracefully across hardware.
pub fn dispatch_matmul<T: Scalar>(
    a: &Matrix<T, DynamicStorage<T>>,
    b: &Matrix<T, DynamicStorage<T>>,
    c: &mut Matrix<T, DynamicStorage<T>>
) -> Result<(), String> {
    
    let backend = select_backend(a.rows(), b.cols(), a.cols());
    
    match backend {
        Architecture::Scalar => {
            // High-level dense MM fallback explicitly mapped
            crate::core::ops::gemm::gemm_cm_unoptimized_xpr(a, b, c)
        },
        Architecture::Simd => {
            // High-level GEMM driver utilizing SIMD/AVX internals
            crate::core::ops::gemm::gemm_cm(a, b, c)
        },
        Architecture::Cuda => {
            #[cfg(feature = "cuda")]
            {
                // Here we attempt to trigger CUDA path explicitly.
                // The current gemm_cm handles GPU transition if 
                // device arrays are prepared, otherwise handles natively.
                crate::core::ops::gemm::gemm_cm(a, b, c)
            }
            #[cfg(not(feature = "cuda"))]
            {
                crate::core::ops::gemm::gemm_cm(a, b, c)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_selection() {
        // Very small matrices should entirely bypass packing overhead
        assert_eq!(select_backend(16, 16, 16), Architecture::Scalar);
        
        // Medium matrices dominate the L1/L2 cache
        assert_eq!(select_backend(256, 256, 256), Architecture::Simd);
        
        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA, even N=1024 uses SIMD
            assert_eq!(select_backend(1024, 1024, 1024), Architecture::Simd);
        }
        
        #[cfg(feature = "cuda")]
        {
            // With CUDA feature engaged, massive matrices offload to GPU
            assert_eq!(select_backend(1024, 1024, 1024), Architecture::Cuda);
        }
    }
}
