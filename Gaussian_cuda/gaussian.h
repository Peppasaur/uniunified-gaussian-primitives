#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include"utils.h"

class Gaussian {
public:
    int id;
    Eigen::Vector3f position=Eigen::Vector3f::Zero();       // 高斯的位置 (3f向量)
    Eigen::Matrix3f orientation=Eigen::Matrix3f::Zero();    // 高斯的朝向 (3x3旋转矩阵)
    Eigen::Vector3f scale=Eigen::Vector3f::Zero();          // 高斯的大小 (3f向量)
    Eigen::Matrix3f scale_mat=Eigen::Matrix3f::Zero();
    Eigen::Matrix3f cov;
    float magnitude=1;
    Bounding_box bbox;

    // 构造函数：从npz文件中的数组进行初始化
    Gaussian() {
    }
    Gaussian(const Eigen::Vector3f& pos, const Eigen::Matrix3f& orient, const Eigen::Vector3f& sca, const Bounding_box bbox,int i,float mat)
        : position(pos), orientation(orient), scale(sca),bbox(bbox),id(i),magnitude(mat) {
            Eigen::Matrix3f scale_mat=Eigen::Matrix3f::Zero();
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

    //float gaussianRayIntegral(double t0,double t1,const Ray& ray);
    //float getGaussian(float t,Ray& ray);
    //bool intersectGaussian(const Ray& ray,float &t0,float &t1);
    Eigen::Vector3f transdir(Eigen::Vector3f dir,Eigen::Matrix3f orien);
    Eigen::Vector3f invtransdir(Eigen::Vector3f dir_tan,Eigen::Matrix3f orien);
    Eigen::Vector3f tan_to_world(Eigen::Vector3f src);
};

// 从npz文件中加载数据并创建Gaussian对象的函数
std::vector<Gaussian> loadGaussiansFromNPZ(const std::string& filename);

/*
inline float Gaussian::gaussianRayIntegral(float t0, float t1,const Ray& ray) {
    // 计算A, B, C
    
    const Eigen::Vector3f& omega=ray.direction;
    const Eigen::Vector3f& pos=this->position;
    const Eigen::Vector3f x=ray.source-pos;
    const Eigen::Matrix3f& Sigma=this->cov;
    const float c=this->magnitude;
    //Eigen::Matrix3f SigmaInv = Sigma.inverse();
    Eigen::Matrix3f scaleInv=Eigen::Matrix3f::Zero();
    scaleInv(0,0)=1/this->scale(0);
    scaleInv(1,1)=1/this->scale(1);
    scaleInv(2,2)=1/this->scale(2);
    Eigen::Matrix3f SigmaInv=this->orientation*scaleInv*scaleInv*(this->orientation.transpose());
    float A = omega.transpose() * SigmaInv * omega;
    float B = (omega.transpose() * SigmaInv * x + x.transpose() * SigmaInv * omega).sum();//problemetic
    float C = x.transpose() * SigmaInv * x;

    // 计算中间变量
    float expFactor = std::exp(B * B / (8 * A) - C / 2);
    
    // 定义tau的积分上下限
    float tau0 = std::sqrt(A / 2) * t0 + B / (2 * std::sqrt(2 * A));
    float tau1 = std::sqrt(A / 2) * t1 + B / (2 * std::sqrt(2 * A));

    // 计算误差函数的差
    float erfDiff = std::erf(tau1) - std::erf(tau0);
    
    float result = expFactor * std::sqrt(M_PI / (2 * A)) * erfDiff;
    if(std::isnan(erfDiff)){
        printf("A%lf B%lf C%lf tau0%f tau1%f\n",A,B,C,tau0,tau1);
        printf("erfDiffnan\n");
        exit(0);
    }
    if(std::isnan(std::sqrt(M_PI / (2 * A)))){
        printf("sqrt\n");
        exit(0);
    }
    if(std::isnan(expFactor)){
        printf("expFactor\n");
        exit(0);
    }
    if(std::isnan(result)){
        printf("result\n");
        exit(0);
    }
    float detSigma = Sigma.determinant();
    
    //printf("detSigma%f\n",detSigma);
    // 计算常数部分 (2 * pi)^(3 / 2)
    float constant = std::pow(2 * M_PI, 3.0 / 2.0);

    // 计算最终结果
    result =result* c / (constant * std::sqrt(detSigma));
    return result;
}
*/
void copyGaussiansToGPU(const std::vector<Gaussian>& gaussians, float*& d_positions, float*& d_orientations, float*& d_scales, float*& d_covs);
#endif
