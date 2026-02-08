#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2f a;
    Eigen::Matrix2f b;
    a << 1, 2, 3, 4;
    b << 10, 20, 30, 40;
    
    Eigen::Matrix2f res = a + b;
    
    for(int i=0; i<res.rows(); ++i) {
        for(int j=0; j<res.cols(); ++j) {
            std::cout << i << "," << j << "," << res(i,j) << std::endl;
        }
    }
    return 0;
}
