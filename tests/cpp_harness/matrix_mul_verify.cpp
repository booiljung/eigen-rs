#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix<float, 2, 3> a;
    Eigen::Matrix<float, 3, 2> b;
    a << 1, 2, 3,
         4, 5, 6;
    b << 7, 8,
         9, 10,
         11, 12;
    
    Eigen::Matrix2f res = a * b;
    
    for(int i=0; i<res.rows(); ++i) {
        for(int j=0; j<res.cols(); ++j) {
            std::cout << i << "," << j << "," << res(i,j) << std::endl;
        }
    }
    return 0;
}
