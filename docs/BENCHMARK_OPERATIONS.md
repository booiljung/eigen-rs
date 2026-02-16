# Benchmark Operations Reference

This document explains the mathematical operations and scenarios covered in the `performance_YYYY-MM-DD_*.md` reports.

## Vector Operations

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **VecDot** | Dot product of two vectors. | $x \cdot y = \sum x_i y_i$ |
| **VecNorm** | Squared Euclidean norm of a vector. | $\|x\|^2 = \sum x_i^2$ |
| **Cross3D** | Cross product of two 3D vectors. | $x \times y$ (3D only) |

## Matrix Arithmetic

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **MatAdd** | Element-wise addition of two dense matrices. | $C = A + B$ |
| **MatMul** | Matrix-Matrix multiplication (GEMM). | $C = A \times B$ |
| **MatScale** | Scalar multiplication of a matrix. | $B = \alpha \cdot A$ |

## Decompositions & Solvers

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **LU** | LU Decomposition with Partial Pivoting. Used for general linear systems. | $PA = LU$ |
| **LLT** | Cholesky Decomposition. Used for symmetric positive-definite systems. | $A = LL^T$ |
| **QR** | QR Decomposition using Householder reflections. Used for least squares. | $A = QR$ |
| **Inverse** | Matrix inversion (typically using LU or Partial Pivoting). | $A^{-1}$ |
| **EigenValues** | Eigenvalue decomposition for self-adjoint (symmetric) matrices. | $A = V D V^{-1}$ |
| **SVD** | Singular Value Decomposition. | $A = U \Sigma V^T$ |

## Sparse Matrix Operations

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **SpMV** | Sparse Matrix-Vector multiplication. $A$ is sparse (CSR/CSC), $x$ is dense. | $y = A x$ |
| **SpMM_Dense** | Sparse Matrix-Dense Matrix multiplication. | $C = A_{sparse} \times B_{dense}$ |

## Geometry

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **QuatMul** | Quaternion multiplication. | $q_1 \otimes q_2$ |

## Benchmark Config Note
- **Size**: Represents the dimension $N$. For Matrix operations, matrices are often $N \times N$.
- **Ratio**: Calculated as `Rust Time / C++ Time`. Lower is better.
    - `< 1.0x`: Rust is faster.
    - `1.0x`: Parity.
    - `> 1.0x`: Rust is slower.
