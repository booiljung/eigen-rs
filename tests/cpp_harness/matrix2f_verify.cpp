#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2f m;
    m << 1, 2,
         3, 4;
    
    // Output in a simple format for Rust to parse: row,col,value
    for(int i=0; i<m.rows(); ++i) {
        for(int j=0; j<m.cols(); ++j) {
            std::cout << i << "," << j << "," << m(i,j) << std::endl;
        }
    }
    return 0;
}
