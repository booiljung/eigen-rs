#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix4f m;
    m << 2, -1, 0, 0,
         -1, 2, -1, 0,
         0, -1, 2, -1,
         0, 0, -1, 2;
    
    auto lu = m.partialPivLu();
    std::cout << "DET," << lu.determinant() << std::endl;
    
    return 0;
}
