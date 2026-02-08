# eigen-rs

[![Korean](https://img.shields.io/badge/Language-Korean-blue.svg)](README.KO.md)

A> **eigen-rs** is a pure Rust port of the **Eigen 3.4** C++ library.

> **Warning**
> This project has **not yet been verified by humans**. All benchmarks and tests are automated. Use with caution in production environments.

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
eigen-rs = "0.1.0"
```

### Example: Matrix Arithmetic

```rust
use eigen_rs::core::matrix::MatrixX;

fn main() {
    let mut a = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    
    // Fill matrices...
    
    let c = &a + &b * 2.0;
    println!("Matrix C:\n{}", c);
}
```

### Example: Solving Linear Systems

```rust
let llt = matrix.llt().expect("Matrix must be positive-definite");
let x = llt.solve(&b).expect("Solve failed");
```

## Documentation

-   [**Benchmarks**](BENCHMARK.md): Methodology and Performance results.
-   [**벤치마크 (Korean)**](BENCHMARK.KO.md): Korean translation of benchmark methodology.
-   [**Roadmap**](ROADMAP.md): Feature parity progress across core modules.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed feature parity progress.

## Acknowledgements

This project is a respectful port of **Eigen**, originally developed by Benoît Jacob and Gaël Guennebaud. `eigen-rs` inherits their dedication to performance and mathematical elegance. We extend our deepest gratitude to the entire Eigen community for establishing the gold standard in C++ linear algebra.

