# The Verification Oracle Methodology: A Blueprint for Porting C++ Libraries to Rust

> **Target Audience**: Future AI Agents & Developers  
> **Purpose**: To provide a rigorous, automated, and gamified strategy for porting complex C++ libraries (e.g., Eigen, PCL, OpenCV) to Rust with 100% confidence.

## 1. Core Philosophy: "Trust but Verify"

When porting a mature C++ library to Rust, the original library is not just a reference—it is the **Oracle**. Your goal is not to "rewrite" logic based on understanding, but to **replicate behavior** based on evidence.

### The Golden Rules
1.  **Total Parity**: The Rust implementation must behave *exactly* like the C++ original for the same inputs, within a defined floating-point error margin.
2.  **Numerical Honesty**: If the C++ library uses a specific algorithm (e.g., column-major storage, specific decomposition steps), the Rust port must honor it unless explicitly deviating for Rust-idiomatic reasons (which must be documented).
3.  **Gamification**: Use a "Perfection Score" (0-100%) to track progress. This motivates completion and makes gaps visible.

---

## 2. Dealing with Differences: Why "Rust Extensions"?

You will inevitably create APIs that do not exist in C++. **This is normal and necessary.**

### Common Reasons for Rust-Only APIs:
1.  **Lifecycle Management**: C++ uses constructors/destructors. Rust needs explicit `new()`, `default()`, `clone()`, `to_owned()`.
2.  **Safety Abstractions**: C++ is happy with raw pointers (`double*`). Rust requires safe wrappers like `as_slice()`, `as_ptr()`, `iter()`.
3.  **Trait Implementations**: Rust idiomatic features like `Display` (`fmt`), `Debug`, or `IntoIterator` require specific methods.
4.  **Helper Accessors**: To test internal state without `friend` classes (C++), Rust often exposes read-only accessors (e.g., `matrix_l()`, `bias()`) that might be hidden or implicit in C++.

**Rule**: All "Rust Extensions" must be verified with **Unit Tests** since they cannot be differential-tested against C++.

---

## 3. The Verification Workflow

Crucially, verification is not an afterthought. It is the driver of development.

### Step 1: Automated API Surface Extraction
Before writing code, map the territory.
-   **Tool**: `extract_cpp_api.py`
-   **Logic**: Parse C++ headers to extract every public class, method, and distinct overload.
-   **Output**: A JSON list of specific functionalities to be ported (e.g., `Matrix::block`, `LLT::solve`).

### Step 2: Implementation & Self-Extraction
As you implement features in Rust:
-   **Tool**: `extract_rust_api.py`
-   **Logic**: Parse your Rust source code (using `syn` or regex) to identify what you *claim* to have implemented.
-   **Output**: A JSON list of your Rust public API.

### Step 3: The Matcher & Gap Analysis
-   **Tool**: `verify_coverage.py` (The Judge)
-   **Logic**: Compare the C++ List vs. Rust List.
    -   **Match**: `inverse()` == `inv()` (Allow standard renaming rules).
    -   **Missing**: Found in C++, missing in Rust.
    -   **Extra**: Found in Rust, missing in C++ (Rust Extensions).
-   **Gamification**: Calculate `Completeness = (Matched / Total C++ APIs) * 100`.

### Step 4: Differential Testing (The "Oracle" Check)
Coverage is meaningless without correctness.
-   **Concept**: Run the *exact same* data through both libraries.
-   **Mechanism**:
    1.  **Generate Data**: Create random matrices/inputs in Rust.
    2.  **Export/FFI**: Send data to a minimal C++ binary linked against the original library.
    3.  **Compute**: C++ library performs the operation.
    4.  **Compare**: Rust asserts its result matches the C++ result ~1e-6 error.
-   **Strict Rule**: A method is only marked `PASS` if it is **explicitly used** in a test file designated for differential testing.

---

## 3. Applying to Future Projects

### Scenario A: Porting PCL (Point Cloud Library)
-   **Oracle**: PCL C++ 1.1x
-   **Data Structures**: `pcl::PointCloud<PointT>`, `pcl::KdTree`
-   **Adaptation**:
    1.  **Extract**: Parse `pcl/point_cloud.h`, `pcl/search/kdtree.h`.
    2.  **Differential Test**: Serialize a `.pcd` file. Load in C++ PCL, run `VoxelGridFilter`. Load in Rust, run implementation. Compare point coordinates.

### Scenario B: Porting OpenCV
-   **Oracle**: OpenCV 4.x
-   **Data Structures**: `cv::Mat`
-   **Adaptation**:
    1.  **Extract**: Parse `opencv2/core.hpp`, `opencv2/imgproc.hpp`.
    2.  **Differential Test**:
        -   Input: A standard image (Lena.jpg).
        -   Op: `cv::Canny` or `cv::warpAffine`.
        -   Compare: Pixel-by-pixel difference between C++ output and Rust output.

---

## 4. The "Completeness Report" Artifact

Generate a `VERIFICATION_REPORT.md` that acts as the source of truth for the project status.

| API Name | C++ Match | Differential Test? | Status |
| :--- | :--- | :--- | :--- |
| `compute()` | `compute()` | ✅ `test_kdtree_diff.rs` | **PASS** |
| `indices()` | `indices()` | ❌ | **FAIL** |
| `to_owned()`| *None* | ✅ `test_internals.rs` | **PASS (Rust Ext)** |

**Final Perfection Score**: **98.5%**

---

> **To the User/AI**: When you start the PCL/OpenCV project, provide this document to the AI. Command it to: *"Build the Verification Toolchain first, based on the `PORTING_STRATEGY.md` file."*
