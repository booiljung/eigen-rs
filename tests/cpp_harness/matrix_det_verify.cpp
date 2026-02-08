#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2f m2;
    m2 << 1, 2,
          3, 4;
    std::cout << "DET2," << m2.determinant() << std::endl;

    Eigen::Matrix3f m3;
    m3 << 1, 2, 3,
          0, 1, 4,
          5, 6, 0;
    // Det: 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
    std::cout << "DET3," << m3.determinant() << std::endl;

    return 0;
}
