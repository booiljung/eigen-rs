#include <Eigen/Dense>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

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

void bench_llt(int size, const std::string& name) {
    MatrixXd A = MatrixXd::Random(size, size);
    A = A * A.transpose(); // Make positive definite
    A.diagonal().array() += 1.0;

    Timer t;
    int iterations = 100;
    for(int i=0; i<iterations; ++i) {
        LLT<MatrixXd> llt(A);
        if(llt.info() != Success) std::cout << "Fail" << std::endl;
    }
    double total_ms = t.elapsed_ms();
    std::cout << "LLT," << name << "," << size << "x" << size << "," << total_ms / iterations << std::endl;
}

void bench_lu(int size, const std::string& name) {
    MatrixXd A = MatrixXd::Random(size, size);
    
    Timer t;
    int iterations = 100;
    for(int i=0; i<iterations; ++i) {
        PartialPivLU<MatrixXd> lu(A);
        // prevent optimization
        if(lu.determinant() > 1e10) std::cout << "";
    }
    double total_ms = t.elapsed_ms();
    std::cout << "LU," << name << "," << size << "x" << size << "," << total_ms / iterations << std::endl;
}

int main() {
    bench_llt(64, "64");
    bench_llt(128, "128");
    bench_llt(256, "256");

    bench_lu(64, "64");
    bench_lu(128, "128");
    bench_lu(256, "256");
    return 0;
}
