#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "/home/qinhaoran/preprocess/cnpy-master/cnpy.h"// 需要安装cnpy库以读取npz文件
#include"utils.h"
#include"ray.h"
#include"../MathLib/FloatStructs.h"

using namespace Selas;
class Gaussian_c {
public:
    int id;
    float3 position=Eigen::Vector3f::Zero();       // 高斯的位置 (3f向量)
    float3x3 orientation=Eigen::Matrix3f::Zero();    // 高斯的朝向 (3x3旋转矩阵)
    float3 scale=Eigen::Vector3f::Zero();          // 高斯的大小 (3f向量)
    float3x3 scale_mat=Eigen::Matrix3f::Zero();
    float3x3 cov;
    float magnitude=1;
    Bounding_box bbox;

    // 构造函数：从npz文件中的数组进行初始化
    Gaussian() {
    }
    Gaussian(const float3& pos, const float3x3& orient, const float3& sca, const Bounding_box bbox,int i)
        : position(pos), orientation(orient), scale(sca),bbox(bbox),id(i) {
            flaot3x3 scale_mat(float3::Zero_,float3::Zero_,float3::Zero_);
            scale_mat(0,0)=scale(0);
            scale_mat(1,1)=scale(1);
            scale_mat(2,2)=scale(2);
            //std::cout<<scale_mat<<std::endl;
            Eigen::Matrix3f mid=orientation*scale_mat;
            cov=mid*(mid.transpose());
        }

    // 打印高斯信息
    void printInfo() const {
        std::cout << "Position: " << position.transpose() << std::endl;
        std::cout << "Orientation:\n" << orientation << std::endl;
        std::cout << "Scale: " << scale.transpose() << std::endl;
    }
    
    bool operator==(const Gaussian& other) const {
        return id == other.id; // ID 相等则视为相等
    }

    float gaussianRayIntegral(float t0,float t1,const Ray& ray);
    float getGaussian(float t,Ray& ray);
    bool intersectGaussian(const Ray& ray,float &t0,float &t1);
    Eigen::Vector3f transdir(Eigen::Vector3f dir);
    Eigen::Vector3f invtransdir(Eigen::Vector3f dir_tan);
};

// 从npz文件中加载数据并创建Gaussian对象的函数
std::vector<Gaussian> loadGaussiansFromNPZ(const std::string& filename);


#endif
