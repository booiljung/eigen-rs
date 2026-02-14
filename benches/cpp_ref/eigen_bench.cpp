#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
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

void bench_vector_ops(int size) {
    Eigen::VectorXf v1 = Eigen::VectorXf::Random(size);
    Eigen::VectorXf v2 = Eigen::VectorXf::Random(size);
    float res = 0;

    int iterations = (size < 1000) ? 10000 : 1000;
    
    // Dot
    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        res += v1.dot(v2);
        v1[0] += 0.00001f;
    }
    auto end = high_resolution_clock::now();
    cout << "VecDot," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;

    // Norm
    start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        res += v1.norm();
        v1[0] += 0.00001f;
    }
    end = high_resolution_clock::now();
    cout << "VecNorm," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
}

void bench_matrix_arithmetic(int size) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf c(size, size);
    
    int iterations = (size < 128) ? 1000 : 100;

    // Add
    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        c.noalias() = a + b;
    }
    auto end = high_resolution_clock::now();
    cout << "MatAdd," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
    
    // Coeff-wise Mul (Schur) - approximating "Scalar Mul" overhead or similar
    // Actually user asked for "Scalar Mul".
    start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        a *= 1.01f;
    }
    end = high_resolution_clock::now();
    cout << "MatScale," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
}

void bench_geometry() {
    int iterations = 10000000;
    
    // Cross Product (3D)
    Eigen::Vector3f v1 = Eigen::Vector3f::Random();
    Eigen::Vector3f v2 = Eigen::Vector3f::Random();
    Eigen::Vector3f v3;
    
    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        v3 = v1.cross(v2);
        v1 += v3 * 0.001f; // dependency
    }
    auto end = high_resolution_clock::now();
    cout << "Cross3D,3," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
    
    // Quaternion Mul
    Eigen::Quaternionf q1 = Eigen::Quaternionf::UnitRandom();
    Eigen::Quaternionf q2 = Eigen::Quaternionf::UnitRandom();
    
    start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
         q1 = q1 * q2;
    }
    end = high_resolution_clock::now();
    cout << "QuatMul,4," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
}

void bench_dense_decomp_extra(int size) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    int iterations = (size < 64) ? 20 : 5;
    
    // LU (PartialPivLU)
    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
         Eigen::PartialPivLU<Eigen::MatrixXf> lu(a);
    }
    auto end = high_resolution_clock::now();
    cout << "LU," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
    
    // QR (HouseholderQR)
    start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
         Eigen::HouseholderQR<Eigen::MatrixXf> qr(a);
    }
    end = high_resolution_clock::now();
    cout << "QR," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
}

void bench_sparse(int size) {
    // Generate sparse matrix (density 0.05)
    Eigen::SparseMatrix<float> sp(size, size);
    float density = 0.05;
    // Fill approximately
    int nnz = static_cast<int>(size * size * density);
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(nnz);
    for(int k=0; k<nnz; ++k) {
        tripletList.push_back(T(rand()%size, rand()%size, 1.0f));
    }
    sp.setFromTriplets(tripletList.begin(), tripletList.end());
    sp.makeCompressed();
    
    Eigen::VectorXf v = Eigen::VectorXf::Random(size);
    Eigen::VectorXf res = Eigen::VectorXf::Zero(size);
    
    int iterations = (size < 256) ? 1000 : 100;
    
    // SpMV
    auto start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        res.noalias() += sp * v;
    }
    auto end = high_resolution_clock::now();
    cout << "SpMV," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
    
    // SpMM (Sparse * Dense)
    Eigen::MatrixXf m = Eigen::MatrixXf::Random(size, 32); // Block of 32 vectors
    Eigen::MatrixXf res_m = Eigen::MatrixXf::Zero(size, 32);
    
    start = high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        res_m.noalias() += sp * m;
    }
    end = high_resolution_clock::now();
    cout << "SpMM_Dense," << size << "," << duration_cast<nanoseconds>(end - start).count() / iterations << endl;
}

int main() {
    // 1. Standard Sweep: 16 to 384, stride 13 (avoid power of 2)
    vector<int> sizes;
    for (int s = 16; s <= 384; s += 13) {
        sizes.push_back(s);
    }
    // Include the standard reference sizes for continuity
    sizes.push_back(64);
    sizes.push_back(128);
    sizes.push_back(256);

    // 2. Small Sweep for heavy ops: 16 to 64, stride 7
    vector<int> small_sizes;
    for (int s = 16; s <= 64; s += 7) {
        small_sizes.push_back(s);
    }
    small_sizes.push_back(32);
    small_sizes.push_back(64);
    
    // Sort and unique to avoid duplicates
    sort(sizes.begin(), sizes.end());
    sizes.erase(unique(sizes.begin(), sizes.end()), sizes.end());
    
    sort(small_sizes.begin(), small_sizes.end());
    small_sizes.erase(unique(small_sizes.begin(), small_sizes.end()), small_sizes.end());
    
    for(int s : sizes) {
        bench_matmul(s);
        bench_llt(s);
        bench_dense_decomp_extra(s);
        bench_matrix_arithmetic(s);
        bench_vector_ops(s); // Standard vector size
        bench_sparse(s);
    }
    
    // 3. Large Vector Benchmarks
    vector<int> large_vec_sizes = {4096, 16384, 65536};
    for(int s : large_vec_sizes) {
        bench_vector_ops(s);
    }

    for(int s : small_sizes) {
        bench_svd(s);
        bench_eigenvalues(s);
    }
    
    bench_geometry();
    
    return 0;
}
