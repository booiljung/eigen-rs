#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(10);

    // 1. Quaternion Multiplication
    Eigen::Quaternionf q1(1.0, 0.1, 0.2, 0.3); // w, x, y, z
    q1.normalize();
    Eigen::Quaternionf q2(1.0, 0.5, 0.4, 0.3);
    q2.normalize();
    auto q_mul = q1 * q2;
    std::cout << "Q_MUL," << q_mul.x() << "," << q_mul.y() << "," << q_mul.z() << "," << q_mul.w() << std::endl;

    // 2. Quaternion Rotation
    Eigen::Vector3f v(1.0, 2.0, 3.0);
    auto v_rot = q1 * v;
    std::cout << "Q_ROT," << v_rot.x() << "," << v_rot.y() << "," << v_rot.z() << std::endl;

    // 3. Slerp
    float t = 0.5f;
    auto q_slerp = q1.slerp(t, q2);
    std::cout << "Q_SLERP," << q_slerp.x() << "," << q_slerp.y() << "," << q_slerp.z() << "," << q_slerp.w() << std::endl;

    // 4. Transform3 (Affine)
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.translate(Eigen::Vector3f(1.0, 2.0, 3.0));
    T.rotate(q1);
    
    Eigen::Vector3f p(10.0, 20.0, 30.0);
    Eigen::Vector3f p_trans = T * p;
    std::cout << "T_TRANS," << p_trans.x() << "," << p_trans.y() << "," << p_trans.z() << std::endl;

    return 0;
}
