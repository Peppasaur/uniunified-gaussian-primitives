#include <Eigen/Dense>
#include <iostream>
#include<cmath>
#include <ctime>
#include"gaussian.h"
#include"ray.h"
#include "cnpy.h"

float Gaussian::getGaussian(float t,Ray& ray){
    Eigen::Vector3f x=ray.source+ray.direction*t;
    Eigen::Vector3f dist=x-this->position;
    Eigen::Matrix3f sigma_inv=cov.inverse();
    float exponent = dist.transpose() * sigma_inv * dist;

    float result=std::exp(-0.5*exponent);
    float detSigma = this->cov.determinant();

    // 计算常数部分 (2 * pi)^(3 / 2)
    float constant = std::pow(2 * M_PI, 3.0 / 2.0);

    // 计算最终结果
    result =result* this->magnitude / (constant * std::sqrt(detSigma));
    return result;
}

bool Gaussian::intersectGaussian(const Ray& ray,float &t_start,float &t_end){
    Eigen::Vector3f oc = ray.source - this->position; // 光线起点到椭球中心的向量
    oc = this->orientation.transpose() * oc; // 旋转到局部坐标系

    Eigen::Vector3f direction = this->orientation.transpose() * ray.direction; // 旋转光线方向

    // 椭球方程参数
    float A = direction.dot(direction) / (this->scale(0) * this->scale(0));
    float B = 2.0 * oc.dot(direction) / (this->scale(0) * this->scale(0));
    float C = (oc.dot(oc) - 1) / (this->scale(0) * this->scale(0));

    // 求解一元二次方程 Ax^2 + Bx + C = 0
    float discriminant = B * B - 4 * A * C; // 判别式

    if (discriminant < 0) {
        // 没有交点
        return false;
    } else {
        // 计算交点
        float sqrtD = std::sqrt(discriminant);
        float t1 = (-B - sqrtD) / (2.0 * A);
        float t2 = (-B + sqrtD) / (2.0 * A);

        // 返回交点时间区间
        t_start = t1 > 0 ? t1 : 0; // t_start
        t_end = t2; // t_end

        return true; // 有交点
    }
}




Eigen::Matrix4f createTransformationMatrix(const Eigen::Vector3f& position, 
                                            const Eigen::Matrix3f& orientation, 
                                            const Eigen::Vector3f& scale 
                                            ) {

    // 创建变换矩阵
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();

    // 设置旋转和缩放
    transformation.block<3, 3>(0, 0) = orientation * scale.asDiagonal();
    
    // 设置平移
    transformation.block<3, 1>(0, 3) = position;

    return transformation;
}


float eigen_implicit_dot(const Eigen::Vector4f& vec) {
    return vec.head<3>().dot(vec.head<3>());  // 计算前三个元素的点积
}

Bounding_box dmnsn_sphere_bounding_fn(const Eigen::Matrix4f& trans,const Eigen::Vector3f& pos,const Eigen::Vector3f& scale,float mat) {
    Bounding_box box;
    

    float sca=std::max(std::max(scale(0),scale(1)),scale(2));
    float exp_bound=0.001;
    float bound_2=-2*log(exp_bound)*pow(sca,2);
    float bound=sqrt(bound_2);
    //bound=5*sca;
    printf("bound%f\n",bound);
    box.min_x=pos(0)-bound;
    box.min_y=pos(1)-bound;
    box.min_z=pos(2)-bound;
    box.max_x=pos(0)+bound;
    box.max_y=pos(1)+bound;
    box.max_z=pos(2)+bound;
    
    /*
    // X 轴方向
    float cx = trans(0, 3); // 取出第0行第3列的平移分量
    float dx = std::sqrt(eigen_implicit_dot(trans.row(0))); // 计算dx
    box.min_x = 2*(cx - dx);
    box.max_x = 2*(cx + dx);

    // Y 轴方向
    float cy = trans(1, 3); // 取出第1行第3列的平移分量
    float dy = std::sqrt(eigen_implicit_dot(trans.row(1))); // 计算dy
    box.min_y = 2*(cy - dy);
    box.max_y = 2*(cy + dy);

    // Z 轴方向
    float cz = trans(2, 3); // 取出第2行第3列的平移分量
    float dz = std::sqrt(eigen_implicit_dot(trans.row(2))); // 计算dz
    box.min_z = 2*(cz - dz);
    box.max_z = 2*(cz + dz);
    */
    return box;
}

std::vector<Gaussian> loadGaussiansFromNPZ(const std::string& filename) {
    
    cnpy::npz_t npz_file = cnpy::npz_load(filename);
    
    // 从npz文件中加载"positions", "orientations", "scales"数组
    cnpy::NpyArray position_data = npz_file["positions"];
    cnpy::NpyArray orientation_data = npz_file["orientations"];
    cnpy::NpyArray scale_data = npz_file["scales"];
    //printf("pos%f",position_data.data<float>()[0]);
    std::vector<Gaussian> gaussians;
    /*
    Eigen::MatrixXf positions = Eigen::Map<Eigen::MatrixXf>(position_data.data<float>(), 3, position_data.shape[0]);
    Eigen::MatrixXf orientations = Eigen::Map<Eigen::MatrixXf>(orientation_data.data<float>(), 9, orientation_data.shape[0]);
    Eigen::MatrixXf scales = Eigen::Map<Eigen::MatrixXf>(scale_data.data<float>(), 3, scale_data.shape[0]);
    */
    float* positions=position_data.data<float>();
    float* orientations=orientation_data.data<float>();
    float* scales=scale_data.data<float>();
    
    int dat=5;
    for (int j = 0; j < position_data.shape[0]; j+=dat) {
        printf("data_sz%d\n",position_data.shape[0]);
        int i=j;
        Eigen::Vector3f position(positions[i*3],positions[i*3+1],positions[i*3+2]);
        //printf("pos%f",position(0));
        Eigen::Matrix3f orientation = Eigen::Map<Eigen::Matrix3f>(orientations+i*9);
        //Eigen::Vector3f scale(30*scales[i*3],30*scales[i*3+1],30*scales[i*3+2]);
        //Eigen::Vector3f scale(130*scales[i*3],130*scales[i*3+1],130*scales[i*3+2]);
        Eigen::Vector3f scale(scales[i*3],scales[i*3+1],scales[i*3+2]);
        //scale*=60;
        scale*=40;
        /*
        if(orientation(1,2)<0){
            //scale=Eigen::Vector3f(0,0,0);
            scale/=1.5;
        }
        */
        printf("orientation%f\n",orientations[i*9+1]);
        std::cout<<scale<<std::endl;
        Eigen::Matrix4f M=createTransformationMatrix(position,orientation,scale);
        //Eigen::Matrix4f S = Eigen::Matrix4f::Identity();
        //S(3, 3) = -1;
        //Eigen::Matrix4f Q=M.transpose()*S*M.inverse();
        //float mat=0.0001;
        float mat=10000000;
        Bounding_box bbox=dmnsn_sphere_bounding_fn(M,position,scale,mat);
        gaussians.emplace_back(position, orientation, scale,bbox,i/dat,mat);
    }
    

    return gaussians;
    
}