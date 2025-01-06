#ifndef RAY_H
#define RAY_H

#include <Eigen/Dense>


class Ray{
public:
    const Eigen::Vector3f source;
    const Eigen::Vector3f direction;
    Ray(const Eigen::Vector3f& src, const Eigen::Vector3f& dir)
        : source(src), direction(dir) {}  // 使用初始化列表来初始化 const 成员
};
#endif