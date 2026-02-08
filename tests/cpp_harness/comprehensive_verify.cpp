#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <iomanip>

// Use high precision for output
void setup_io() {
    std::cout << std::fixed << std::setprecision(10);
}

void verify_vec_add(int size) {
    Eigen::VectorXf a(size);
    Eigen::VectorXf b(size);
    for(int i=0; i<size; ++i) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }
    Eigen::VectorXf c = a + b;
    for(int i=0; i<size; ++i) {
        std::cout << "ADD," << i << "," << c[i] << std::endl;
    }
}

void verify_vec_sub(int size) {
    Eigen::VectorXf a(size);
    Eigen::VectorXf b(size);
    for(int i=0; i<size; ++i) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }
    Eigen::VectorXf c = a - b;
    for(int i=0; i<size; ++i) {
        std::cout << "SUB," << i << "," << c[i] << std::endl;
    }
}

void verify_vec_mul(int size) {
    Eigen::VectorXf a(size);
    Eigen::VectorXf b(size);
    for(int i=0; i<size; ++i) {
        a[i] = (float)i;
        b[i] = 2.0f;
    }
    // Scalar mul equivalent check
    Eigen::VectorXf c = a * 2.0f; 
    for(int i=0; i<size; ++i) {
        std::cout << "MUL," << i << "," << c[i] << std::endl;
    }
}

void verify_vec_functions(int size) {
    Eigen::VectorXf a(size);
    for(int i=0; i<size; ++i) {
        a[i] = (float)i * 0.1f;
    }
    
    // SIN
    Eigen::VectorXf s = a.array().sin();
    for(int i=0; i<size; ++i) {
        std::cout << "SIN," << i << "," << s[i] << std::endl;
    }

    // EXP
    Eigen::VectorXf e = a.array().exp();
    for(int i=0; i<size; ++i) {
        std::cout << "EXP," << i << "," << e[i] << std::endl;
    }
    
    // SQRT (avoid negative)
    Eigen::VectorXf sq = a.array().abs().sqrt();
    for(int i=0; i<size; ++i) {
        std::cout << "SQRT," << i << "," << sq[i] << std::endl;
    }
}

void verify_gemm(int rows, int cols, int depth) {
    // C = A * B
    // A: rows x depth
    // B: depth x cols
    Eigen::MatrixXf a(rows, depth);
    Eigen::MatrixXf b(depth, cols);
    
    for(int i=0; i<rows; ++i) {
        for(int k=0; k<depth; ++k) {
            a(i, k) = (float)(i + k) * 0.1f;
        }
    }
    
    for(int k=0; k<depth; ++k) {
        for(int j=0; j<cols; ++j) {
            b(k, j) = (float)(k - j) * 0.1f;
        }
    }
    
    Eigen::MatrixXf c = a * b;
    
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            std::cout << "GEMM," << i << "," << j << "," << c(i, j) << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    setup_io();
    
    int vec_size = 128; 
    
    if (argc > 1) {
        vec_size = std::atoi(argv[1]);
    }

    verify_vec_add(vec_size);
    verify_vec_sub(vec_size);
    verify_vec_mul(vec_size);
    verify_vec_functions(vec_size);
    
    // Small GEMM for correctness verification
    verify_gemm(16, 16, 16);
    
    return 0;
}
