#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <complex>

int main() {
    // 4x4 complex matrices A and B
    Eigen::Matrix4cd A, B;
    A << std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 0.0), std::complex<double>(0.0, 1.0), std::complex<double>(0.5, 0.5),
         std::complex<double>(0.5, 0.5), std::complex<double>(3.0, 2.0), std::complex<double>(1.0, -1.0), std::complex<double>(0.1, 0.2),
         std::complex<double>(1.0, 0.0), std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0), std::complex<double>(0.3, 0.4),
         std::complex<double>(0.2, 0.3), std::complex<double>(0.4, 0.5), std::complex<double>(0.6, 0.1), std::complex<double>(1.5, 1.2);
    
    B << std::complex<double>(5.0, 0.0), std::complex<double>(1.0, 1.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.1, 0.1),
         std::complex<double>(1.0, -1.0), std::complex<double>(4.0, 2.0), std::complex<double>(2.0, 1.0), std::complex<double>(0.2, 0.3),
         std::complex<double>(3.0, 0.0), std::complex<double>(1.0, 0.0), std::complex<double>(5.0, -1.0), std::complex<double>(0.5, 0.5),
         std::complex<double>(0.1, 0.2), std::complex<double>(0.2, 0.1), std::complex<double>(0.3, 0.4), std::complex<double>(2.0, 1.0);
    
    // Generalized eigenvalues are eigenvalues of A * B.inverse()
    // This assumes B is invertible, which it is for our test data.
    Eigen::Matrix4cd C = A * B.inverse();
    Eigen::ComplexEigenSolver<Eigen::Matrix4cd> solver(C);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "C++ ComplexEigenSolver failed" << std::endl;
        return 1;
    }
    
    const auto& eigenvalues = solver.eigenvalues();

    std::cout << std::fixed << std::setprecision(12);
    
    std::cout << "VALS_ROWS," << eigenvalues.rows() << std::endl;
    for(int i = 0; i < eigenvalues.rows(); ++i) {
        std::cout << "VAL," << i << "," << eigenvalues(i).real() << "," << eigenvalues(i).imag() << std::endl;
    }
    
    return 0;
}
