# Geometry Module Report

This document details the mapping between `Eigen/src/Geometry` and `eigen-rs/src/core/geometry`.

## 1. Directory Mapping

| C++ Eigen (`Eigen/src/Geometry`) | Rust `eigen-rs` (`src/core/geometry`) | Description |
|:---|:---|:---|
| `Geometry/` | `core/geometry/` | Geometric transformations |

## 2. File & Class Mapping

| C++ File | C++ Class | Rust File | Rust Struct/Trait | Status | Notes |
|:---|:---|:---|:---|:---|:---|
| `Quaternion.h` | `Quaternion<T>` | `core/geometry/quaternion.rs` | `Quaternion<T>` | **Full** | Unit quaternion ops |
| `Transform.h` | `Transform<T, Dim, Mode>` | `core/geometry/transform.rs` | `Transform<T, D, M>` | **Full** | Affine/Isometry/Projective |
| `Translation.h` | `Translation<T, Dim>` | `core/geometry/translation.rs` | `Translation<T, D>` | **Full** | |
| `Scaling.h` | `Scaling` (Uniform/Non-uniform) | `core/geometry/scaling.rs` | `Scaling` | **Full** | |
| `EulerAngles.h` | `eulerAngles()` | `core/geometry/euler_angles.rs` | `euler_angles()` | **Full** | XYZ, ZYZ, etc. |

## 3. Testing Strategy

- **Unit Tests**:
    - `tests/geometry_test.rs`: Basic Quaternion arithmetic.
    - `tests/geometry_transformations_test.rs`: Composition of Rotation * Translation * Scaling.
- **Verification**:
    - `tests/cpp_harness/geometry_verify.cpp`: Verifies quaternion multiplication and rotation application against C++.

## 4. Hardware Acceleration

- **SIMD**:
    - Quaternion multiplication uses vector dot products optimized via `Packet`.
    - Transform application ($T \times v$) uses GEMV/GEMM optimized paths.
