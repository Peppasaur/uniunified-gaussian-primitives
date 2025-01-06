#include <Eigen/Dense>
#include <iostream>
#include<cmath>
#include <ctime>
#include <cuda_runtime.h>
#include"gaussian.h"
#include"ray.h"
#include "cnpy.h"
#define CHECK_CUDA_ERROR(call)                                     \
    {                                                              \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__            \
                      << " at line " << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;  \
            exit(err);                                            \
        }                                                          \
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
    float exp_bound=0.001/mat;
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
    
    int dat=1;
    Eigen::Matrix3f orientation1;
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
        if(j==0)orientation1=orientation;
        std::cout<<orientation1<<std::endl;
        Eigen::Matrix4f M=createTransformationMatrix(position,orientation,scale);
        //Eigen::Matrix4f S = Eigen::Matrix4f::Identity();
        //S(3, 3) = -1;
        //Eigen::Matrix4f Q=M.transpose()*S*M.inverse();
        //float mat=0.0001;
        float mat=0.001;
        Bounding_box bbox=dmnsn_sphere_bounding_fn(M,position,scale,mat);
        gaussians.emplace_back(position, orientation, scale,bbox,i/dat,mat);
    }
    

    return gaussians;
    
}


void copyGaussiansToGPU(const std::vector<Gaussian>& gaussians, float*& d_positions, float*& d_orientations, float*& d_scales, float*& d_covs) {
    int h_data = 5; 
    int* d_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, 3*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));

    int count = gaussians.size();
    // 在 host 端为每个成员变量创建连续的数组
    float* h_positions = new float[count * 3];    // 每个 position 是 3 个 float
    float* h_orientations = new float[count * 9]; // 每个 orientation 是 3x3 的矩阵 (9个 float)
    float* h_scales = new float[count * 3];       // 每个 scale 是 3 个 float
    float* h_covs = new float[count * 9];         // 每个 cov 是 3x3 的矩阵 (9个 float)

    // 将 Gaussian 对象中的 position、orientation、scale 和 cov 分别填充到对应的数组中
    for (int i = 0; i < count; ++i) {
        // 复制 position
        for (int j = 0; j < 3; ++j) {
            h_positions[i * 3 + j] = gaussians[i].position(j);
            h_scales[i * 3 + j] = gaussians[i].scale(j);
        }

        // 复制 orientation
        for (int j = 0; j < 9; ++j) {
            h_orientations[i * 9 + j] = gaussians[i].orientation(j / 3, j % 3);
            h_covs[i * 9 + j] = gaussians[i].cov(j / 3, j % 3);
        }
    }
    printf("222\n");
    // 为 GPU 端分配连续的内存空间
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_positions, count * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_orientations, count * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_scales, count * 3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_covs, count * 9 * sizeof(float)));
    printf("333\n");
    // 将打包后的数组复制到 GPU 上
    CHECK_CUDA_ERROR(cudaMemcpy(d_positions, h_positions, count * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_orientations, h_orientations, count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_scales, h_scales, count * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_covs, h_covs, count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    
    // 释放 host 端内存
    delete[] h_positions;
    delete[] h_orientations;
    delete[] h_scales;
    delete[] h_covs;
}
