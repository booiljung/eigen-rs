#include <iostream>
#include <Eigen/Sparse>
#include <iomanip>
#include <vector>

int main() {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletListA;
    tripletListA.push_back(T(0, 0, 1.0));
    tripletListA.push_back(T(1, 1, 2.0));
    tripletListA.push_back(T(0, 2, 3.0));

    Eigen::SparseMatrix<double, Eigen::RowMajor> A(2, 3);
    A.setFromTriplets(tripletListA.begin(), tripletListA.end());

    std::vector<T> tripletListB;
    tripletListB.push_back(T(0, 0, 10.0));
    tripletListB.push_back(T(0, 1, 5.0));
    tripletListB.push_back(T(1, 1, 1.0));

    Eigen::SparseMatrix<double, Eigen::RowMajor> B(2, 3);
    B.setFromTriplets(tripletListB.begin(), tripletListB.end());

    Eigen::SparseMatrix<double, Eigen::RowMajor> C_add = A + B;
    Eigen::SparseMatrix<double, Eigen::RowMajor> C_sub = A - B;
    Eigen::SparseMatrix<double, Eigen::RowMajor> C_scale = A * 2.5;
    Eigen::SparseMatrix<double, Eigen::RowMajor> C_transpose = A.transpose();

    // Sparse-Sparse Multiplication
    // A (2x3), B_for_mul (3x2)
    std::vector<T> tripletListB2;
    tripletListB2.push_back(T(0, 0, 10.0));
    tripletListB2.push_back(T(0, 1, 5.0));
    tripletListB2.push_back(T(2, 0, 1.0));
    Eigen::SparseMatrix<double, Eigen::RowMajor> B2(3, 2);
    B2.setFromTriplets(tripletListB2.begin(), tripletListB2.end());
    Eigen::SparseMatrix<double, Eigen::RowMajor> C_mul = A * B2;

    std::cout << std::fixed << std::setprecision(12);

    auto printSparse = [](const std::string& name, const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat) {
        std::cout << name << "_ROWS," << mat.rows() << std::endl;
        std::cout << name << "_COLS," << mat.cols() << std::endl;
        std::cout << name << "_NNZ," << mat.nonZeros() << std::endl;
        for (int k=0; k<mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it) {
                std::cout << name << "_VAL," << it.row() << "," << it.col() << "," << it.value() << std::endl;
            }
        }
    };

    printSparse("ADD", C_add);
    printSparse("SUB", C_sub);
    printSparse("SCALE", C_scale);
    printSparse("TRANSPOSE", C_transpose);
    printSparse("MUL", C_mul);

    return 0;
}
