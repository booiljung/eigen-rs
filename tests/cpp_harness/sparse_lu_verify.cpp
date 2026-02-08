#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

using namespace Eigen;

int main() {
    std::cout << std::setprecision(18);
    int n = 10;
    
    // Create a non-symmetric sparse matrix
    SparseMatrix<double> a(n, n);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = (double)((i * 7 + j * 3) % 11) - 5.0;
            if (val != 0.0 && (i + j) % 2 == 0) {
                triplets.push_back(Triplet<double>(i, j, val));
            }
        }
        // Ensure non-singular by adding to diagonal
        triplets.push_back(Triplet<double>(i, i, 10.0));
    }
    a.setFromTriplets(triplets.begin(), triplets.end());
    a.makeCompressed();

    SparseLU<SparseMatrix<double>> solver;
    solver.compute(a);
    if (solver.info() != Success) {
        std::cerr << "Factorization failed" << std::endl;
        return 1;
    }

    // Solve Ax = b
    VectorXd b = VectorXd::Random(n);
    VectorXd x = solver.solve(b);

    // Output B
    std::cout << "B_SIZE " << b.size() << std::endl;
    for (int i = 0; i < b.size(); ++i) {
        std::cout << "B_VAL " << i << " " << b(i) << std::endl;
    }

    // Output partial results
    // SparseLU in Eigen is supernodal, P * A * Q = L * U.
    // Our simplicial implementation is P * A = L * U (Q = I).
    // Wait... if I want to compare, I should use a version of SparseLU that is simplicial.
    // But Eigen's SparseLU is always supernodal.
    // SimplicialLU in Eigen? No, SparseLU is the only one.
    
    // However, if we don't enable column ordering, Q might be Identity.
    // But PartialPivoting is still different.
    
    // Let's just compare the final solution x for parity.
    std::cout << "X_SIZE " << x.size() << std::endl;
    for (int i = 0; i < x.size(); ++i) {
        std::cout << "X_VAL " << i << " " << x(i) << std::endl;
    }

    return 0;
}
