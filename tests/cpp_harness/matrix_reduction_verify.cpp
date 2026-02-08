#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix<float, 2, 2> m;
    m << 1.5, 2.5,
         3.5, 4.5;
    
    std::cout << "SUM," << m.sum() << std::endl;
    std::cout << "MIN," << m.minCoeff() << std::endl;
    std::cout << "MAX," << m.maxCoeff() << std::endl;
    std::cout << "MEAN," << m.mean() << std::endl;
    
    return 0;
}
