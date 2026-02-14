## Project Overview

- **Objective**: Porting the C++ **Eigen 3.4** library to Rust.
- **Project Name**: **`eigen-rs`**
- **Acknowledgement**: This project respects the achievements of the original [Eigen](https://eigen.tuxfamily.org/) library and inherits its spirit.

## Tech Stack

- **Language**: **Rust 1.75+**
- **Build System**: **Cargo** (Build/Dependency Management)
- **Optimization**:
  - **SIMD** (x86_64, ARM64 support)
  - **CUDA 11.0+** (GPU acceleration, maintaining system-independent compatibility)
- **Infrastructure**: **GitHub Actions** (CI/CD)
- **Temporary Files**: All temporary files, logs, and benchmark binaries must be placed in the `tmp/` directory (which is git-ignored) to keep the project root clean.

## Development Rules

### 0. Operation & Ethics
- **SemVer Compliance**: Strictly follow Semantic Versioning to protect backward compatibility.
- **Transparency**: Clearly state implemented and unimplemented features to avoid user confusion.
- **Contributor Respect**: Maintain clear guides and technical respect for open-source contributors.

### 1. Code Style
- **Rustfmt**: Must pass `cargo fmt`.
- **Clippy**: Must fix all warnings (`cargo clippy`).
- **Comments**: Write doc comments for all public APIs.
- **Error Handling**:
  - Do not use `unwrap()`, `expect()`.
  - Principle of explicit handling based on `Result`, `Option`.
- **Ownership Management**:
  - Minimize unnecessary `clone()`.
  - Prioritize utilizing Borrowing and Lifetimes.

### 2. Testing
- **Full Scope Testing**: Include unit tests for all features.
- **Differential Testing**:
  - Compare results of C++ Eigen and Rust implementation for the same input in real-time.
  - **Rationale**: To guarantee numerical honesty by checking floating-point operation results, which may vary slightly by system/compiler environment, in the same environment (Adopting Runtime C++ Harness approach).
- **Modularization**: Implement within `#[cfg(test)]`.
- **Flexibility**: Provide automatic fallback when SIMD/CUDA environments are absent.
- **CI Integration**: Branch merge is possible only when `cargo test` passes.

### 3. Documentation
- **README.md**: Include overview, installation, examples.
- **CHANGELOG.md**: Manage change history.
- **API Documentation**: Keep up-to-date with `cargo doc`.

### 4. External Source Management
- **Submodule Usage**: When bringing in external source code (original Eigen, etc.), must use `git submodule add` instead of `git clone`.

### 5. Performance Management
- **Measurement Tool**: Perform precise benchmarks using the `criterion` library.
- **Optimized Build**: For fair performance evaluation of hardware acceleration (SIMD), always measure in `RUSTFLAGS="-C target-cpu=native"` environment.
- **Result Recording**: Check `docs/reports/` for detailed logs.
- **Target Value**: Aim to maintain performance within **1.5x** of Eigen for core operations (GEMM, Decompositions).

## Core Design Principles (Advanced)

### 1. Zero-Cost Abstraction
- **Lazy Evaluation**: Implement Expression Templates utilizing Rust Traits/GATs to prevent intermediate temporary object creation.
- **Memory Alignment**: Manage aligned memory allocation in 16/32/64-byte units for SIMD optimization.
- **Const Generics**: Elegant integration of Fixed-size and Dynamic-size matrices.

### 2. Numerical Stability & Extensibility
- **Scalar Generics**: Support `f32`, `f64`, `Complex`, and user-defined numeric types.
- **Numerical Honesty**: Clarify the allowable error range (epsilon) for operations, and warn the user when precision is sacrificed.
- **Benchmark**: Build infrastructure for objective performance comparison with original C++ Eigen using `criterion`.
- **Test Coverage**: Maintain and manage high code coverage utilizing `tarpaulin`, etc.
- **License**: Review MPL 2.0 policy to inherit the spirit of the original work.

## CI/CD
- **Automation**: Build/Test based on GitHub Actions.
- **Trigger**: Run on main branch push and PR creation.
- **Merge Condition**: CI pass required.

## 6. Benchmark Methodology
Please refer to [BENCHMARK.md](BENCHMARK.md) for the detailed benchmark methodology and developer guide.
