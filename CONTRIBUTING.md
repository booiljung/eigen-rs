# Contributing to eigen-rs

Thank you for your interest in contributing to **eigen-rs**! This project is a pure Rust port of the Eigen C++ library, aiming for high performance and API compatibility.

## Branching Strategy

We follow a **Trunk-Based Development** model tailored for a porting project.

### 1. Main Branch (`main`)
-   **Role**: Production-ready, **human-verified** code.
-   **Status**: Strictly stable.
-   **Update Rule**: Only merges from `develop` are allowed after **human review**.
-   **CI/CD**: Tags on this branch trigger Crates.io publication.

### 2. Develop Branch (`develop`)
-   **Role**: The integration branch for all new features and ports.
-   **Status**: Compilable and passes automated CI tests.
-   **Target**: All Pull Requests (PRs) should target this branch.
-   **Note**: Contains code that passes `simd_verify_test` but may await human audit.

### 3. Release Tags (`vX.Y.Z`)
-   **Role**: Immutable snapshots for Crates.io releases.
-   **Format**: `v[Major].[Minor].[Patch]` (e.g., `v3.4.0`).

### 3. Feature Branches (`feature/*`)
-   **Role**: Implementation of new modules or major features.
-   **Naming**: `feature/[module_name]` (e.g., `feature/sparse-qr`, `feature/geometry`).
-   **Workflow**: Branch off `main` -> Implement -> PR to `main`.

### 4. Bugfix Branches (`fix/*`)
-   **Role**: Fixes for bugs or compilation errors.
-   **Naming**: `fix/[issue_description]` (e.g., `fix/gemm-overflow`).

### 5. Chore Branches (`chore/*`)
-   **Role**: Documentation, CI/CD configuration, or refactoring without logic changes.

---

## Contribution Workflow

1.  **Fork & Clone**: Fork the repository and clone it locally.
2.  **Create Branch**:
    ```bash
    git checkout -b feature/my-awesome-feature
    ```
3.  **Implement**: Write code following the "Development Rules" in `GEMINI.md`.
    -   **Style**: Run `cargo fmt` and `cargo clippy`.
    -   **Test**: Ensure `cargo test` passes.
4.  **Verify**:
    -   If touching math kernels, run `cargo test --test simd_verify_test` to verify against C++ Eigen.
5.  **Push & PR**: Push to your fork and open a Pull Request to `eigen-rs/main`.

## CI/CD Checks
Your PR will be automatically tested by GitHub Actions for:
-   Formatting (`rustfmt`)
-   Lints (`clippy`)
-   Tests (`cargo test`)
-   Cross-Verification (`simd_verify_test`)

## Releasing (Maintainers Only)
To publish a new version:
1.  Update `version` in `Cargo.toml`.
2.  Commit and Push to `main`.
3.  Create a **GitHub Release** with tag `vX.Y.Z`.
4.  The `publish.yml` workflow will automatically publish to crates.io.
