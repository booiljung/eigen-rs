#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main() {
    int N = 100;
    MatrixXf A(N, N);
    MatrixXf B(N, N);

    // Deterministic Initialization
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            A(i, j) = (float)(i + j);
            B(i, j) = (float)(i - j);
        }
    }

    MatrixXf C = A * B;

    // Output Result
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            std::cout << i << "," << j << "," << C(i, j) << "\n";
        }
    }

    return 0;
}
