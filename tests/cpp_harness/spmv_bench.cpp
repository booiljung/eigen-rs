#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace Eigen;
using namespace std;

// Simple timer (same as gemm_bench)
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

void bench_spmv(int rows, int cols, double density, const std::string& name) {
    SparseMatrix<double> A(rows, cols);
    A.reserve(VectorXi::Constant(cols, rows * density));
    
    // Random fill
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(int k=0; k<cols; ++k) {
        for(int i=0; i<rows; ++i) {
            if(dist(gen) < density) {
                A.insert(i, k) = dist(gen);
            }
        }
    }
    A.makeCompressed();

    VectorXd x = VectorXd::Random(cols);
    VectorXd y(rows);

    // Warmup
    y.noalias() = A * x;

    Timer t;
    int iterations = 100;
    for(int i=0; i<iterations; ++i) {
        y.noalias() = A * x;
        if (y(0) > 1e10) std::cout << ""; 
    }
    double total_ms = t.elapsed_ms();
    double avg_ms = total_ms / iterations;

    std::cout << "SpMV," << name << "," << rows << "x" << cols << "," << avg_ms << std::endl;
}

int main() {
    // 1000x1000, 1% density (Matching Rust bench)
    bench_spmv(1000, 1000, 0.01, "1000_1pct");
    
    // 4096, 0.1%
     bench_spmv(4096, 4096, 0.001, "4096_0.1pct");

    return 0;
}
