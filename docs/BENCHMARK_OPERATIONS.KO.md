# 벤치마크 연산 설명 (Benchmark Operations Reference)

`performance_YYYY-MM-DD_*.md` 리포트에 포함된 주요 수학적 연산과 시나리오에 대한 설명입니다.

## 벡터 연산 (Vector Operations)

| 연산 (Operation) | 설명 (Description) | 수식 / 로직 (Formula) |
| :--- | :--- | :--- |
| **VecDot** | 두 벡터의 내적 (Dot Product). | $x \cdot y = \sum x_i y_i$ |
| **VecNorm** | 벡터의 유클리드 노름의 제곱 (Squared Norm). | $\|x\|^2 = \sum x_i^2$ |
| **Cross3D** | 두 3차원 벡터의 외적 (Cross Product). | $x \times y$ (3D only) |

## 행렬 산술 연산 (Matrix Arithmetic)

| 연산 (Operation) | 설명 (Description) | 수식 / 로직 (Formula) |
| :--- | :--- | :--- |
| **MatAdd** | 두 밀집 행렬(Dense Matrix)의 원소별 덧셈. | $C = A + B$ |
| **MatMul** | 행렬 곱셈 (Matrix Multiplication, GEMM). | $C = A \times B$ |
| **MatScale** | 행렬의 스칼라 곱 (Scalar Multiplication). | $B = \alpha \cdot A$ |

## 분해 및 솔버 (Decompositions & Solvers)

| 연산 (Operation) | 설명 (Description) | 수식 / 로직 (Formula) |
| :--- | :--- | :--- |
| **LU** | 부분 피벗팅(Partial Pivoting)을 사용한 LU 분해. 일반 선형 시스템 풀이에 사용. | $PA = LU$ |
| **LLT** | 촐레스키 분해 (Cholesky Decomposition). 대칭 양의 정부호(SPD) 행렬에 사용. | $A = LL^T$ |
| **QR** | 하우스홀더 반사(Householder Reflection)를 사용한 QR 분해. 최소자승법 등에 사용. | $A = QR$ |
| **Inverse** | 행렬 역행렬 구하기 (주로 LU 분해 활용). | $A^{-1}$ |
| **EigenValues** | 자기 수반(Self-Adjoint, 대칭) 행렬의 고유값 분해. | $A = V D V^{-1}$ |
| **SVD** | 특이값 분해 (Singular Value Decomposition). | $A = U \Sigma V^T$ |
| **LDLT** | 피벗팅을 사용한 로버스트 촐레스키 분해. | $A = P^T L D L^T P$ |
| **Hessenberg** | 헤센버그 분해. 일반 행렬을 헤센버그 형태로 축소. | $A = Q H Q^T$ |
| **Tridiagonal** | 삼중대각화. 자기 수반 행렬을 삼중대각 형태로 축소. | $A = Q T Q^T$ |
| **GeneralizedEigen** | 일반화된 자기 수반 고유값 문제 솔버. | $Ax = \lambda Bx$ |
| **RealSchur** | 실수 슈어(Schur) 분해. | $A = U T U^T$ |
| **Determinant** | 행렬식(Determinant) 계산 (주로 LU 분해 활용). | $\det(A)$ |

## 희소 행렬 연산 (Sparse Matrix Operations)

| 연산 (Operation) | 설명 (Description) | 수식 / 로직 (Formula) |
| :--- | :--- | :--- |
| **SpMV** | 희소 행렬-벡터 곱셈. $A$는 희소 행렬(CSR/CSC), $x$는 밀집 벡터. | $y = A x$ |
| **SpMM_Dense** | 희소 행렬-밀집 행렬 곱셈. | $C = A_{sparse} \times B_{dense}$ |
| **SparseLU** | 희소 LU 솔버. | 희소 LU를 사용한 $Ax = b$ |
| **SparseQR** | 희소 QR 솔버. | 희소 QR을 사용한 $Ax = b$ |
| **SparseView** | 밀집 행렬을 희소 행렬로 변환 (압축 기반). | `Dense -> Sparse` |

## 기하학 (Geometry)

| 연산 (Operation) | 설명 (Description) | 수식 / 로직 (Formula) |
| :--- | :--- | :--- |
| **QuatMul** | 쿼터니언(Quaternion) 곱셈. | $q_1 \otimes q_2$ |
| **Transform** | 아핀 변환(Affine Transformation) 곱셈. | $T \times v$ |
| **Translation** | 이동(Translation) 적용. | $Tr \times v$ |
| **Scaling** | 균일 스케일링(Scaling) 적용. | $S \times v$ |
| **AngleAxis** | 축-각(Angle-Axis) 회전을 회전 행렬로 변환. | `AngleAxis -> Mat3` |
| **EulerAngles** | 회전 행렬을 오일러 각(Euler Angles)으로 변환. | `Mat3 -> Euler` |

## 벤치마크 설정 참고 (Benchmark Config Note)
- **Size**: 차원 $N$을 의미합니다. 행렬 연산의 경우 주로 $N \times N$ 행렬입니다.
- **Ratio (비율)**: `Rust 시간 / C++ 시간`으로 계산됩니다. 수치가 낮을수록 Rust가 더 빠름을 의미합니다.
    - `< 1.0x`: Rust가 더 빠름 (Faster).
    - `1.0x`: 동등 수준 (Parity).
    - `> 1.0x`: Rust가 더 느림 (Slower).
