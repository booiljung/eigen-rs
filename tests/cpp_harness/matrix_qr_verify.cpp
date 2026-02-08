#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix<float, 3, 2> m;
    m << 1, 2,
         3, 4,
         5, 6;
    
    auto qr = m.householderQr();
    Eigen::MatrixXf q = qr.householderQ();
    Eigen::MatrixXf r = qr.matrixQR().triangularView<Eigen::Upper>();
    
    std::cout << "Q_ROWS," << q.rows() << std::endl;
    std::cout << "Q_COLS," << q.cols() << std::endl;
    for(int i=0; i<q.rows(); ++i) {
        for(int j=0; j<q.cols(); ++j) {
            std::cout << "Q," << i << "," << j << "," << q(i,j) << std::endl;
        }
    }
    
    std::cout << "R_ROWS," << r.rows() << std::endl;
    std::cout << "R_COLS," << r.cols() << std::endl;
    for(int i=0; i<r.rows(); ++i) {
        for(int j=0; j<r.cols(); ++j) {
            std::cout << "R," << i << "," << j << "," << r(i,j) << std::endl;
        }
    }
    
    return 0;
}
