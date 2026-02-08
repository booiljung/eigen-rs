# Decompositions Roadmap

**Status**: 100% Complete

## Dense Matrix Decompositions
- **Linear System Solvers**
    - [x] **LU**: `PartialPivLU` (Row pivoting), `FullPivLU` (Rank-revealing).
    - [x] **QR**: `HouseholderQR`, `ColPivHouseholderQR`, `FullPivHouseholderQR`.
    - [x] **Cholesky**: `LLT` (LL^T), `LDLT` (LDL^T, robust semi-definite).
    - [x] **Orthogonal**: `CompleteOrthogonalDecomposition`.
- **Eigenvalues & SVD**
    - [x] **SVD**: `JacobiSVD` (High precision), `BDCSVD` (Divide & Conquer - Pending).
    - [x] **SelfAdjointEigenSolver**: High-performance solver for symmetric/Hermitian matrices.
    - [x] **Complex/General Eigensolvers**: `EigenSolver` (Real) [x], `ComplexEigenSolver` [x], `GeneralizedEigenSolver` [x].
    - [x] **Hessenberg/Schur**: `HessenbergDecomposition` [x], `RealSchur` [x], `ComplexSchur` [x].
