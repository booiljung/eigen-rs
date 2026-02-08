# Core Module Roadmap

**Status**: 100% Complete

## Architecture & Dense Linear Algebra
- **Dimension & Metaprogramming**
    - [x] **Const Generics**: Unified `Fixed` and `Dynamic` size handling.
    - [x] **Unrolling**: Template-based unrolling for small fixed-size matrices (2x2 to 4x4).
    - [x] **Evaluator Design**: `Evaluator` trait pattern for decoupling expressions from storage.
- **Expression Template Engines**
    - [x] **Lazy Evaluators**: `CwiseUnaryOp`, `CwiseBinaryOp`, `CwiseNullaryOp`, `CwiseTernaryOp`.
    - [x] **Aliasing Handling**: `noalias()`, `eval()`, `nestByValue()`.
    - [x] **Loop Fusion**: Automatic fusion of coefficient-wise operations.
- **Memory & Storage Policies**
    - [x] **Alignment**: Strict 16/32/64-byte alignment (SSE/AVX/AVX-512).
    - [x] **Storage Classes**: `DenseStorage`, `Array`, `Matrix`.
    - [x] **Existing Buffers**: `Map`, `Ref`, `Stride` (Inner/Outer/Dynamic).
- **API Surface**
    - [x] **Slicing**: `block()`, `row()`, `col()`, `topLeftCorner()`, `bottomRightCorner()`.
    - [x] **Advanced Indexing**: `seq()`, `seqN()`, `IndexedView`, `Reshaped`.
    - [x] **Reductions**: `sum()`, `prod()`, `minCoeff()`, `maxCoeff()`, `norm()`, `hypot()`.
    - [x] **Booleans**: `all()`, `any()`, `count()`.
