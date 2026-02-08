#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace Eigen;
using namespace std;

// Simple timer
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

template<typename T>
void bench_gemm(int size, const std::string& name) {
    Matrix<T, Dynamic, Dynamic> A = Matrix<T, Dynamic, Dynamic>::Random(size, size);
    Matrix<T, Dynamic, Dynamic> B = Matrix<T, Dynamic, Dynamic>::Random(size, size);
    Matrix<T, Dynamic, Dynamic> C(size, size);

    // Warmup
    C.noalias() = A * B;

    Timer t;
    int iterations = 10;
    if (size < 256) iterations = 100;
    if (size > 1024) iterations = 5;

    for(int i=0; i<iterations; ++i) {
        C.noalias() = A * B;
        // black box to prevent optimization (not perfect in C++ but ok for basic bench)
        if (C(0,0) > 1e10) std::cout << ""; 
    }
    double total_ms = t.elapsed_ms();
    double avg_ms = total_ms / iterations;

    std::cout << "GEMM," << name << "," << size << "x" << size << "," << avg_ms << std::endl;
}

int main() {
    // f32 benchmarks
    bench_gemm<float>(64, "f32");
    bench_gemm<float>(256, "f32");
    bench_gemm<float>(512, "f32");
    bench_gemm<float>(1024, "f32");

    // f64 benchmarks
    bench_gemm<double>(64, "f64");
    bench_gemm<double>(256, "f64");
    bench_gemm<double>(512, "f64");
    // bench_gemm<double>(1024, "f64"); // Optional if slow

    return 0;
}
