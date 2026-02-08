#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

int main() {
    // 4x4 Symmetric matrix
    Eigen::Matrix4f m;
    m << 10.0, 1.0, 2.0, 3.0,
          1.0, 20.0, 4.0, 5.0,
          2.0, 4.0, 30.0, 6.0,
          3.0, 5.0, 6.0, 40.0;
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> solver(m);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "C++ Solver failed" << std::endl;
        return 1;
    }
    
    const auto& eigenvalues = solver.eigenvalues();
    const auto& eigenvectors = solver.eigenvectors();

    std::cout << std::fixed << std::setprecision(10);
    
    std::cout << "VALS_ROWS," << eigenvalues.rows() << std::endl;
    for(int i = 0; i < eigenvalues.rows(); ++i) {
        std::cout << "VAL," << i << "," << eigenvalues(i) << std::endl;
    }
    
    std::cout << "VECS_ROWS," << eigenvectors.rows() << std::endl;
    std::cout << "VECS_COLS," << eigenvectors.cols() << std::endl;
    for(int i = 0; i < eigenvectors.rows(); ++i) {
        for(int j = 0; j < eigenvectors.cols(); ++j) {
            std::cout << "VEC," << i << "," << j << "," << eigenvectors(i, j) << std::endl;
        }
    }
    
    return 0;
}
