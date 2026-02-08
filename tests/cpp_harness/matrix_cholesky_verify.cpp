#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

int main() {
    // 3x3 SPD matrix for LLT
    Eigen::Matrix3f m;
    m << 4, 12, -16,
         12, 37, -43,
         -16, -43, 98;
    
    Eigen::LLT<Eigen::Matrix3f> llt(m);
    Eigen::Matrix3f l = llt.matrixL();
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "LLT_L_ROWS,3" << std::endl;
    std::cout << "LLT_L_COLS,3" << std::endl;
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            std::cout << "LLT_L," << i << "," << j << "," << l(i,j) << std::endl;
        }
    }

    // Solve Ax = b where b = [1, 2, 3]T
    Eigen::Vector3f b(1, 2, 3);
    Eigen::Vector3f x = llt.solve(b);
    for(int i=0; i<3; ++i) {
        std::cout << "LLT_SOLVE," << i << "," << x(i) << std::endl;
    }

    // 3x3 Symmetric matrix for LDLT
    Eigen::Matrix3f m3;
    m3 << 4, 12, -16,
          12, 37, -43,
          -16, -43, 98;
    Eigen::LDLT<Eigen::Matrix3f> ldlt(m3);
    Eigen::Matrix3f L = ldlt.matrixL();
    Eigen::Vector3f D = ldlt.vectorD();
    auto P = ldlt.transpositionsP();
    
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
             std::cout << "LDLT_L," << i << "," << j << "," << L(i,j) << std::endl;
        }
        std::cout << "LDLT_D," << i << "," << D(i) << std::endl;
    }
    // Output transpositions
    for(int i=0; i<P.size(); ++i) {
        std::cout << "LDLT_P," << i << "," << P.indices()(i) << std::endl;
    }

    return 0;
}
