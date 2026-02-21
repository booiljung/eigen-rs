# Sparse Module Roadmap

**Status**: 100% Complete

## Sparse Linear Algebra
- **Core Sparse**
    - [x] **Storage**: `SparseMatrix` (CSR/CSC), `SparseVector`, `CompressedStorage`.
    - [x] **Evaluators**: Iterators over non-zero elements (`InnerIterator`).
    - [x] **Arithmetic**: Algebraic operations (`+`, `-`, `*scalar`, `transpose`).
    - [x] **Multiplication**: Sparse-Sparse Matrix Product ($C = A \cdot B$).
- **Direct & Iterative Solvers**
    - [x] **Built-in**: `SimplicialLLT` [x], `SimplicialLDLT` [x], `SparseLU` [x], `SparseQR` [x].
    - [x] **Iterative**: `ConjugateGradient`, `BiCGSTAB`, `GMRES`.
    - [x] **Preconditioners**: `IncompleteCholesky`, `IncompleteLUT`.
- **External Interfaces**
    - [x] **Bridges**: LAPACK (LU/QR optimized), Cholmod, UmfPack, SuperLU, Pardiso, Pastix, SuiteSparse.
