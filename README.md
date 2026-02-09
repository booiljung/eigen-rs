# eigen-rs

[![CI](https://github.com/booiljung/eigen-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/booiljung/eigen-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/eigen-rs.svg)](https://crates.io/crates/eigen-rs)
[![Docs.rs](https://docs.rs/eigen-rs/badge.svg)](https://docs.rs/eigen-rs)
[![License](https://img.shields.io/crates/l/eigen-rs.svg)](https://github.com/booiljung/eigen-rs/blob/main/LICENSE)
[![Korean](https://img.shields.io/badge/Language-Korean-blue.svg)](README.KO.md)

A> **eigen-rs** is a pure Rust port of the **Eigen 3.4** C++ library.

> **Warning**
> This project was **generated and verified by AI**. While it has passed comprehensive automated tests and benchmarks against C++ Eigen, it has **not yet been verified by humans**. Use with caution in production environments.

`eigen-rs` aims to provide the same numerical stability, performance, and elegant API as the original Eigen library, leveraging Rust's safety and modern features like Const Generics and Trait-based SIMD/CUDA dispatching.

## Contributing
Interested in contributing? Please read our [Contributing Guide](CONTRIBUTING.md) for details on our branching strategy and development workflow.

## Features

- **Header-only Spirit**: Lean and optimized for Zero-Cost abstraction.
- **Lazy Evaluation**: Advanced expression template system to minimize temporary allocations.
- **Hardware Acceleration**: 
    - **SIMD**: SSE, AVX, and NEON support via architecture-agnostic traits.
    - **CUDA**: Automatic GPU offloading for large-scale arithmetic expressions.
- **Numerical Decompositions**: Robust solvers including LU, QR, Cholesky (LLT/LDLT), and JacobiSVD.
- **Geometry Module**: Quaternions and linear transformations for robotics and computer graphics.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
eigen-rs = "3.4.0"
```

### Example: Matrix Arithmetic

```rust
use eigen_rs::core::matrix::MatrixX;

fn main() {
    let mut a = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    
    // Fill matrices...
    *a.get_mut(0, 0).unwrap() = 1.0;
    *b.get_mut(0, 0).unwrap() = 2.0;
    
    // Perform operations
    // Note: Expression templates like (&a + &b * 2.0) are supported but concise assignment 
    // to a new Matrix needs explicit evaluation or assignment methods.
    // Here we show a step-by-step approach for clarity:
    let mut b_scaled = b.clone(); 
    b_scaled.scale(2.0);

    let mut c = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    c.assign(&(&a + &b_scaled)).unwrap();
    
    println!("Matrix C:\n{:?}", c);
}
```

### Example: Solving Linear Systems

```rust
use eigen_rs::core::matrix::Matrix3;

fn main() {
    let matrix = Matrix3::<f32>::identity();
    let b = Matrix3::<f32>::identity(); // Placeholder
    let llt = matrix.llt().expect("Matrix must be positive-definite");
    let x = llt.solve(&b).expect("Solve failed");
}
```

## Documentation

-   [**Project Guide & Benchmarks**](GEMINI.md): Project overview and Benchmark Methodology.
-   [**프로젝트 가이드 (Korean)**](GEMINI.KO.md): Project overview and Benchmark Methodology (Korean).
-   [**Roadmap**](ROADMAP.md): Feature parity progress across core modules.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed feature parity progress.

## Acknowledgements

This project is a respectful port of **Eigen**, originally developed by Benoît Jacob and Gaël Guennebaud. `eigen-rs` inherits their dedication to performance and mathematical elegance. We extend our deepest gratitude to the entire Eigen community for establishing the gold standard in C++ linear algebra.

