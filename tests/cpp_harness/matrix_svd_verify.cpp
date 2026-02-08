#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

int main() {
    Eigen::Matrix<float, 3, 2> m;
    m << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    
    // Use Thin SVD for now to match current Rust implementation of One-Sided Jacobi
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    const auto& s = svd.singularValues();
    const auto& u = svd.matrixU();
    const auto& v = svd.matrixV();

    std::cout << std::fixed << std::setprecision(10);
    
    for(int i=0; i<s.size(); ++i) {
        std::cout << "S," << i << "," << s(i) << std::endl;
    }
    
    for(int i=0; i<u.rows(); ++i) {
        for(int j=0; j<u.cols(); ++j) {
            std::cout << "U," << i << "," << j << "," << u(i,j) << std::endl;
        }
    }
    
    for(int i=0; i<v.rows(); ++i) {
        for(int j=0; j<v.cols(); ++j) {
            std::cout << "V," << i << "," << j << "," << v(i,j) << std::endl;
        }
    }
    
    return 0;
}
