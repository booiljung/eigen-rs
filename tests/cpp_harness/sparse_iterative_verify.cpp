#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;

int main() {
    std::cout << std::setprecision(18);
    int n = 20;
    
    // 1. Conjugate Gradient (SPD)
    SparseMatrix<double> a_spd(n, n);
    std::vector<Triplet<double>> triplets_spd;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                triplets_spd.push_back(Triplet<double>(i, j, 20.0 + (i % 5)));
            } else if (std::abs(i - j) == 1) {
                triplets_spd.push_back(Triplet<double>(i, j, -1.0));
            }
        }
    }
    a_spd.setFromTriplets(triplets_spd.begin(), triplets_spd.end());
    
    VectorXd b_cg = VectorXd::Random(n);
    ConjugateGradient<SparseMatrix<double>, Lower|Upper, DiagonalPreconditioner<double>> cg;
    cg.setTolerance(1e-12);
    cg.compute(a_spd);
    VectorXd x_cg = cg.solve(b_cg);

    std::cout << "CG_B_SIZE " << b_cg.size() << std::endl;
    for (int i = 0; i < b_cg.size(); ++i) std::cout << "CG_B_VAL " << i << " " << b_cg(i) << std::endl;
    std::cout << "CG_X_SIZE " << x_cg.size() << std::endl;
    for (int i = 0; i < x_cg.size(); ++i) std::cout << "CG_X_VAL " << i << " " << x_cg(i) << std::endl;

    // 2. BiCGSTAB (General)
    SparseMatrix<double> a_gen(n, n);
    std::vector<Triplet<double>> triplets_gen;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = (double)((i * 13 + j * 7) % 17) - 8.0;
            if (val != 0.0 && std::abs(i - j) <= 3) {
                triplets_gen.push_back(Triplet<double>(i, j, val));
            }
        }
        triplets_gen.push_back(Triplet<double>(i, i, 50.0)); // Ensure diagonal dominance for easy convergence
    }
    a_gen.setFromTriplets(triplets_gen.begin(), triplets_gen.end());

    VectorXd b_bicg = VectorXd::Random(n);
    BiCGSTAB<SparseMatrix<double>, DiagonalPreconditioner<double>> bicg;
    bicg.setTolerance(1e-12);
    bicg.compute(a_gen);
    VectorXd x_bicg = bicg.solve(b_bicg);

    std::cout << "BICG_B_SIZE " << b_bicg.size() << std::endl;
    for (int i = 0; i < b_bicg.size(); ++i) std::cout << "BICG_B_VAL " << i << " " << b_bicg(i) << std::endl;
    std::cout << "BICG_X_SIZE " << x_bicg.size() << std::endl;
    for (int i = 0; i < x_bicg.size(); ++i) std::cout << "BICG_X_VAL " << i << " " << x_bicg(i) << std::endl;

    return 0;
}
