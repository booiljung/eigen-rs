#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using namespace Eigen;

int main() {
    std::cout << std::setprecision(18);
    int n = 10;
    // Create a simple SPD sparse matrix: A = M * M^T + I
    SparseMatrix<double> m(n, n);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((i + j) % 3 == 0) {
                triplets.push_back(Triplet<double>(i, j, (double)(i + j + 1) / 10.0));
            }
        }
    }
    m.setFromTriplets(triplets.begin(), triplets.end());
    
    SparseMatrix<double> a = m * m.transpose();
    for (int i = 0; i < n; ++i) {
        a.coeffRef(i, i) += 1.0; // Ensure positive definite
    }
    a.makeCompressed();

    // SimplicialLLT with NaturalOrdering to match Rust
    SimplicialLLT<SparseMatrix<double>, Lower, NaturalOrdering<int>> llt;
    llt.compute(a);
    if (llt.info() != Success) {
        std::cerr << "Factorization failed" << std::endl;
        return 1;
    }

    // Solve Ax = b
    VectorXd b = VectorXd::Random(n);
    VectorXd x = llt.solve(b);

    // Output B vector
    std::cout << "B_SIZE " << b.size() << std::endl;
    for (int i = 0; i < b.size(); ++i) {
        std::cout << "B_VAL " << i << " " << b(i) << std::endl;
    }

    // Output L matrix
    SparseMatrix<double> l = llt.matrixL();
    std::cout << "L_NONZEROS " << l.nonZeros() << std::endl;
    for (int k = 0; k < l.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(l, k); it; ++it) {
            std::cout << "L_VAL " << it.row() << " " << it.col() << " " << it.value() << std::endl;
        }
    }

    // Output solution x
    std::cout << "X_SIZE " << x.size() << std::endl;
    for (int i = 0; i < x.size(); ++i) {
        std::cout << "X_VAL " << i << " " << x(i) << std::endl;
    }

    return 0;
}
