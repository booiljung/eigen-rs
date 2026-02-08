#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <complex>

int main() {
    // 4x4 complex matrix
    Eigen::Matrix4cd m;
    m << std::complex<double>(0.35, 0.45), std::complex<double>(0.45, -0.14), std::complex<double>(-0.14, 0.25), std::complex<double>(-0.17, 0.11),
         std::complex<double>(0.09, 0.07), std::complex<double>(0.07, 0.35), std::complex<double>(-0.54, -0.13), std::complex<double>(0.35, 0.17),
         std::complex<double>(-0.44, -0.33), std::complex<double>(-0.33, 0.11), std::complex<double>(-0.03, 0.17), std::complex<double>(0.17, 0.09),
         std::complex<double>(0.25, -0.32), std::complex<double>(-0.32, 0.09), std::complex<double>(-0.13, 0.07), std::complex<double>(0.11, 0.11);
    
    Eigen::ComplexSchur<Eigen::Matrix4cd> schur(m);
    
    if (schur.info() != Eigen::Success) {
        std::cerr << "C++ ComplexSchur failed" << std::endl;
        return 1;
    }
    
    const auto& T = schur.matrixT();
    const auto& U = schur.matrixU();

    std::cout << std::fixed << std::setprecision(12);
    
    std::cout << "T_ROWS," << T.rows() << std::endl;
    std::cout << "T_COLS," << T.cols() << std::endl;
    for(int i = 0; i < T.rows(); ++i) {
        for(int j = 0; j < T.cols(); ++j) {
            std::cout << "T," << i << "," << j << "," << T(i, j).real() << "," << T(i, j).imag() << std::endl;
        }
    }
    
    std::cout << "U_ROWS," << U.rows() << std::endl;
    std::cout << "U_COLS," << U.cols() << std::endl;
    for(int i = 0; i < U.rows(); ++i) {
        for(int j = 0; j < U.cols(); ++j) {
            std::cout << "U," << i << "," << j << "," << U(i, j).real() << "," << U(i, j).imag() << std::endl;
        }
    }
    
    return 0;
}
