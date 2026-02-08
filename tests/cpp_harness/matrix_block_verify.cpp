#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix<float, 4, 4> m;
    m << 1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12,
         13, 14, 15, 16;
    
    // Block(1, 1, 2, 2)
    auto b = m.block(1, 1, 2, 2);
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            std::cout << "BLOCK," << i << "," << j << "," << b(i,j) << std::endl;
        }
    }

    // Row 2
    auto r = m.row(2);
    for(int j=0; j<4; ++j) {
        std::cout << "ROW," << j << "," << r(0,j) << std::endl;
    }

    // Col 3
    auto c = m.col(3);
    for(int i=0; i<4; ++i) {
        std::cout << "COL," << i << "," << c(i,0) << std::endl;
    }

    return 0;
}
