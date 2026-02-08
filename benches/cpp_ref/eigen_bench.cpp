#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <string>

using namespace std;
using namespace std::chrono;

void bench_matmul(int size) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf c(size, size);

    int iterations = (size < 128) ? 100 : 20;
    
    // Warm up
    for(int i=0; i<5; ++i) {
        c.noalias() = a * b;
    }

    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        c.noalias() = a * b;
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "MatMul," << size << "," << duration / iterations << endl;
}

void bench_llt(int size) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf m = a * a.transpose() + Eigen::MatrixXf::Identity(size, size) * size;
    
    int iterations = (size < 128) ? 100 : 20;

    // Warm up
    for(int i=0; i<5; ++i) {
        Eigen::LLT<Eigen::MatrixXf> llt(m);
    }

    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        Eigen::LLT<Eigen::MatrixXf> llt(m);
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "LLT," << size << "," << duration / iterations << endl;
}

void bench_svd(int size) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    int iterations = (size < 64) ? 20 : 5;

    // Warm up
    for(int i=0; i<2; ++i) {
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "SVD," << size << "," << duration / iterations << endl;
}

void bench_eigenvalues(int size) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf m = a + a.transpose();
    int iterations = (size < 64) ? 20 : 5;

    // Warm up
    for(int i=0; i<2; ++i) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(m);
    }

    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(m);
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "EigenValues," << size << "," << duration / iterations << endl;
}

int main() {
    vector<int> sizes = {64, 128, 256};
    vector<int> small_sizes = {32, 64};
    
    for(int s : sizes) {
        bench_matmul(s);
    }
    for(int s : sizes) {
        bench_llt(s);
    }
    for(int s : small_sizes) {
        bench_svd(s);
    }
    for(int s : small_sizes) {
        bench_eigenvalues(s);
    }
    
    return 0;
}
