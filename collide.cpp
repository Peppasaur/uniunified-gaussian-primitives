#include <iostream>
#include <memory> 
#include "fcl/math/constants.h"
#include "fcl/narrowphase/collision.h"
#include "fcl/narrowphase/collision_object.h"
#include "fcl/narrowphase/distance.h"

using namespace fcl;

// 创建椭球的函数
std::shared_ptr<CollisionGeometry<double>> createEllipsoid(double rx, double ry, double rz) {
    return std::make_shared<Ellipsoid<double>>(rx, ry, rz);
}

// 创建 OBB 的函数
std::shared_ptr<CollisionGeometry<double>> createBox(const Vector3<double>& half_extents) {
    return std::make_shared<Box<double>>(half_extents[0] * 2, half_extents[1] * 2, half_extents[2] * 2);  // Box 需要宽度、高度和深度，而不是半边长
}

// 检测椭球和 OBB 是否相交的函数
bool checkEllipsoidBoxCollision(const Vector3<double>& ellipsoid_center, double rx, double ry, double rz, 
                                 const Vector3<double>& box_center, const Vector3<double>& box_half_extents, 
                                 const Matrix3<double>& box_rotation) {
    // 创建椭球和 Box 的几何体
    detail::GJKSolver_libccd<double> solver;
    Box<double> box1(2.0,2.0,2.0);
    Ellipsoid<double> ellipsoid1(2, 2, 2);    

    auto ellipsoid = createEllipsoid(rx, ry, rz);
    auto box = createBox(box_half_extents);

    // 设置椭球的变换
    Transform3<double> ellipsoid_transform;
    ellipsoid_transform.setIdentity();
    ellipsoid_transform.translation() = ellipsoid_center;
    //ellipsoid_transform.linear() = box_rotation; 

    // 设置 Box 的变换
    Transform3<double> box_transform;
    box_transform.setIdentity();
    box_transform.translation() = box_center;
    //box_transform.linear() = box_rotation;  // 设置 Box 的旋转

    bool overlap_ellipsoid = solver.shapeIntersect(box1, box_transform, ellipsoid1, ellipsoid_transform, nullptr);
    std::cout<<overlap_ellipsoid<<std::endl;
    OBB<double> obb2;
    computeBV(ellipsoid1, ellipsoid_transform, obb2);
    std::cout << "Width: " << obb2.width() << std::endl;
    std::cout << "Height: " << obb2.height() << std::endl;
    std::cout << "Depth: " << obb2.depth() << std::endl;
    fcl::Vector3<double> center = obb2.center();
    std::cout << "Center: (" << center.x() << ", " << center.y() << ", " << center.z() << ")" << std::endl;

    std::cout << "Box Representation: " << box1.representation(2) << std::endl;

    std::vector<fcl::Vector3<double>> vertices = box1.getBoundVertices(box_transform);

    // 输出顶点
    std::cout << "Box Vertices:" << std::endl;
    for (const auto& vertex : vertices) {
        std::cout << "Vertex: (" << vertex[0] << ", " << vertex[1] << ", " << vertex[2] << ")" << std::endl;
    }
    
    // 创建碰撞对象
    CollisionObject<double> ellipsoid_object(ellipsoid, ellipsoid_transform);
    CollisionObject<double> box_object(box, box_transform);

    // 碰撞检测请求和结果
    CollisionRequest<double> request;
    CollisionResult<double> result;
    
    // 执行碰撞检测
    collide(&ellipsoid_object, &box_object, request, result);

    // 返回是否发生碰撞
    return result.isCollision();
}
/*
int main() {
    // 椭球中心和半轴长度
    Vector3<double> ellipsoid_center(0, -2.9, 0);
    double rx = 2.0, ry = 2.0, rz = 2.0;

    // Box 中心、半边长和旋转矩阵
    Vector3<double> box_center(0, 0, 100);
    Vector3<double> box_half_extents(1.0, 1.0, 1.0);
    Matrix3<double> box_rotation;  // Box 旋转矩阵
    box_rotation << 1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;  // 无旋转（单位矩阵）

    // 检测椭球和 Box 是否相交
    if (checkEllipsoidBoxCollision(ellipsoid_center, rx, ry, rz, box_center, box_half_extents, box_rotation)) {
        std::cout << "椭球和 Box 碰撞。" << std::endl;
    } else {
        std::cout << "椭球和 Box 没有碰撞。" << std::endl;
    }

    return 0;
}
*/
