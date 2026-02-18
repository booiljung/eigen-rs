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
| **LDLT** | Robust Cholesky Decomposition with pivoting. | $A = P^T L D L^T P$ |
| **Hessenberg** | Hessenberg Decomposition. Reduces general matrix to Hessenberg form. | $A = Q H Q^T$ |
| **Tridiagonal** | Tridiagonalization of self-adjoint matrices. | $A = Q T Q^T$ |
| **GeneralizedEigen** | Generalized Self-Adjoint Eigen Solver. | $Ax = \lambda Bx$ |
| **RealSchur** | Real Schur Decomposition. | $A = U T U^T$ |
| **Determinant** | Matrix Determinant calculation (typically via LU). | $\det(A)$ |

## Sparse Matrix Operations

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **SpMV** | Sparse Matrix-Vector multiplication. $A$ is sparse (CSR/CSC), $x$ is dense. | $y = A x$ |
| **SpMM_Dense** | Sparse Matrix-Dense Matrix multiplication. | $C = A_{sparse} \times B_{dense}$ |
| **SparseLU** | Sparse LU Solver. | $Ax = b$ using sparse LU |
| **SparseQR** | Sparse QR Solver. | $Ax = b$ using sparse QR |
| **SparseView** | Conversion from Dense Matrix to Sparse Matrix (Compressed Base). | `Dense -> Sparse` |

## Geometry

| Operation | Description | Formula / Logic |
| :--- | :--- | :--- |
| **QuatMul** | Quaternion multiplication. | $q_1 \otimes q_2$ |
| **Transform** | Affine Transformation multiplication. | $T \times v$ |
| **Translation** | Translation application. | $Tr \times v$ |
| **Scaling** | Uniform Scaling application. | $S \times v$ |
| **AngleAxis** | Angle-Axis rotation to Rotation Matrix conversion. | `AngleAxis -> Mat3` |
| **EulerAngles** | Rotation Matrix to Euler Angles conversion. | `Mat3 -> Euler` |

## Benchmark Config Note
- **Size**: Represents the dimension $N$. For Matrix operations, matrices are often $N \times N$.
- **Ratio**: Calculated as `Rust Time / C++ Time`. Lower is better.
    - `< 1.0x`: Rust is faster.
    - `1.0x`: Parity.
    - `> 1.0x`: Rust is slower.
