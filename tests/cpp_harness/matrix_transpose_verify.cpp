#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix<float, 2, 3> m;
    m << 1, 2, 3,
         4, 5, 6;
    
    auto res = m.transpose();
    
    for(int i=0; i<res.rows(); ++i) {
        for(int j=0; j<res.cols(); ++j) {
            std::cout << i << "," << j << "," << res(i,j) << std::endl;
        }
    }
    return 0;
}
