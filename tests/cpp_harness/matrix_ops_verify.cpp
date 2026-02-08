#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(10);

    Eigen::Matrix2f a;
    a << 10.0, 20.0,
         30.0, 40.0;

    Eigen::Matrix2f b;
    b << 1.0, 2.0,
         3.0, 4.0;

    // Subtraction
    std::cout << "SUB" << std::endl;
    Eigen::Matrix2f sub = a - b;
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            std::cout << i << "," << j << "," << sub(i,j) << std::endl;
        }
    }

    // Scalar Multiplication
    std::cout << "MUL" << std::endl;
    Eigen::Matrix2f mul = a * 2.5f;
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            std::cout << i << "," << j << "," << mul(i,j) << std::endl;
        }
    }

    return 0;
}
