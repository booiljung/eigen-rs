#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2f m2;
    m2 << 1, 2,
          3, 4;
    auto inv2 = m2.inverse();
    std::cout << "INV2,0,0," << inv2(0,0) << std::endl;
    std::cout << "INV2,0,1," << inv2(0,1) << std::endl;
    std::cout << "INV2,1,0," << inv2(1,0) << std::endl;
    std::cout << "INV2,1,1," << inv2(1,1) << std::endl;

    Eigen::Matrix4f m4;
    m4 << 2, -1, 0, 0,
         -1, 2, -1, 0,
         0, -1, 2, -1,
         0, 0, -1, 2;
    auto inv4 = m4.inverse();
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            std::cout << "INV4," << i << "," << j << "," << inv4(i,j) << std::endl;
        }
    }

    return 0;
}
