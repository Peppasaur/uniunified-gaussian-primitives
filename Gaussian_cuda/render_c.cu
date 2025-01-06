#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include<cmath>
#include <cstring> 
#include"time.h"
#include <Eigen/Dense>
#include <curand_kernel.h>
#include"render_c.h"
#include"gaussian_bbox.h"
#include <cuda_runtime.h>

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

__global__ void setupKernel(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 使用种子 seed 初始化每个线程的 cuRAND 状态
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ void matrixMultiply(const double* matA, const double* matB, double* result) {
    // 手动计算矩阵乘法，假设矩阵是3x3并且使用行主序存储
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                result[i * 3 + j] += matA[i * 3 + k] * matB[k * 3 + j];
            }
        }
    }
}

__device__ void matrixMultiply1(const float* matA, const float* matB, float* result) {
    // 手动计算矩阵乘法，假设矩阵是3x3并且使用行主序存储
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                result[i * 3 + j] += matA[i * 3 + k] * matB[k * 3 + j];
            }
        }
    }
}

__device__ void matrixVectorMultiply(const double* mat, const double* vec, double* result) {
    for (int i = 0; i < 3; ++i) {
        result[i] = mat[i * 3 + 0] * vec[0] +
                    mat[i * 3 + 1] * vec[1] +
                    mat[i * 3 + 2] * vec[2];
    }
}

__device__ void matrixVectorMultiply1(const float* mat, const float* vec, float* result) {
    for (int i = 0; i < 3; ++i) {
        result[i] = mat[i * 3 + 0] * vec[0] +
                    mat[i * 3 + 1] * vec[1] +
                    mat[i * 3 + 2] * vec[2];
    }
}

__device__ void matrixTranspose(const double* mat, double* result) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i * 3 + j] = mat[j * 3 + i];
        }
    }
}

__device__ void matrixTranspose1(const float* mat, float* result) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i * 3 + j] = mat[j * 3 + i];
        }
    }
}


__device__ void printMatrix3d(const double* matrix) {
    // 打印矩阵的各个元素，逐行打印
    //printf("Matrix3d:\n");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%f ", matrix[i*3+j]);
        }
        printf("\n"); // 换行
    }
}

__device__ Eigen::Vector3f intersectRayWithPlane(Ray &ray,float* d_positions,int gid,
                                              const Eigen::Vector3f& planeNormal) {
    Eigen::Vector3f rayOrigin(ray.source.x,ray.source.y,ray.source.z);
    Eigen::Vector3f rayDir(ray.direction.x,ray.direction.y,ray.direction.z);
    Eigen::Vector3f pointOnPlane(d_positions[3*gid],d_positions[3*gid+1],d_positions[3*gid+2]);
    float denominator = planeNormal.dot(rayDir);
    
    // 如果点积为0，说明射线与平面平行或在平面上
    /*
    if (std::abs(denominator) < 1e-6) {
        return Eigen::Vec; // 没有交点
    }
    */

    // 计算参数 t
    float t = planeNormal.dot(pointOnPlane - rayOrigin) / denominator;
    
    // 如果 t < 0，说明交点在射线的反方向上
    /*
    if (t < 0) {
        return NULL; // 射线与平面没有交点（射线背面）
    }
    */

    // 计算交点 P = O + t * D
    Eigen::Vector3f intersectionPoint = rayOrigin + t * rayDir;
    return intersectionPoint;
}

__device__ float intersectRayWithPlane_t(Ray &ray,float* d_positions,int gid,
                                              const Eigen::Vector3f& planeNormal) {
    Eigen::Vector3f rayOrigin(ray.source.x,ray.source.y,ray.source.z);
    Eigen::Vector3f rayDir(ray.direction.x,ray.direction.y,ray.direction.z);
    Eigen::Vector3f pointOnPlane(d_positions[3*gid],d_positions[3*gid+1],d_positions[3*gid+2]);
    float denominator = planeNormal.dot(rayDir);
    // 计算参数 t
    float t = planeNormal.dot(pointOnPlane - rayOrigin) / denominator;
    return t;
}

__device__ float gaussianRayIntegral2(float* &positions, float* &orientations, float* &scales, float* &covs, float* magnitude, int &gid ,float& t0, float& t1, Ray& ray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Eigen::Vector3f norm(orientations[9*gid+2],orientations[9*gid+5],orientations[9*gid+8]);
    Eigen::Vector3f ray_dir(ray.direction.x, ray.direction.y, ray.direction.z);
    float dot = norm.transpose() * (-ray_dir);
    /*
    if(dot<0){
        return 0;
        //printf("-dot\n");
    }
    */
    Eigen::Matrix3f orient;
    orient<<orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                                orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                                orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];
    Eigen::Vector3f pos(positions[gid*3],positions[gid*3+1],positions[gid*3+2]);
    Eigen::Vector3f point=intersectRayWithPlane(ray,positions,gid,norm);
    point=point-pos;
    Eigen::Vector3f loc_point=orient.transpose()*point;
    //printf("loc_pointz%f\n",loc_point(2));
    float scalex=scales[3*gid],scaley=scales[3*gid+1];
    //printf("scx%f scy%f\n",scalex,scaley);
    float normalization_factor = 1.0f / (2.0f * M_PI * scalex * scaley);
    float exponent_x = - (loc_point(0) * loc_point(0)) / (2.0f * scalex*scalex);
    float exponent_y = - (loc_point(1) * loc_point(1)) / (2.0f * scaley*scaley);
    /*
    if(exponent_x<-150){
        printf("exponent_x\n");
    }
    */
    float result =normalization_factor * expf(exponent_x + exponent_y);
    if (isnan(result)) {
        printf("nanresult\n");
    }
    if (isinf(result)) {
        printf("infresult\n");
    }
    //if(result>0.0000001)printf("result %f\n",result);
    return result*magnitude[gid]*0.001;
    //return result*100;
}

__device__ float gaussianRayIntegral3(float* &positions, float* &orientations, float* &scales, float* &covs, float* magnitude, int &gid ,float3 point3) {
    Eigen::Matrix3f orient;
    orient<<orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                                orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                                orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];
    Eigen::Vector3f pos(positions[gid*3],positions[gid*3+1],positions[gid*3+2]);
    Eigen::Vector3f point(point3.x,point3.y,point3.z);
    point=point-pos;
    Eigen::Vector3f loc_point=orient.transpose()*point;
    //printf("loc_pointz%f\n",loc_point(2));
    float scalex=scales[3*gid],scaley=scales[3*gid+1];
    //printf("scx%f scy%f\n",scalex,scaley);
    float normalization_factor = 1.0f / (2.0f * M_PI * scalex * scaley);
    float exponent_x = - (loc_point(0) * loc_point(0)) / (2.0f * scalex*scalex);
    float exponent_y = - (loc_point(1) * loc_point(1)) / (2.0f * scaley*scaley);
    /*
    if(exponent_x<-150){
        printf("exponent_x\n");
    }
    */
    float result =normalization_factor * expf(exponent_x + exponent_y);
    if (isnan(result)) {
        printf("nanresult\n");
    }
    if (isinf(result)) {
        printf("infresult\n");
    }
    //if(result>0.0000001)printf("result %f\n",result);
    return result*magnitude[gid]*0.001;
    //return result*100;
}
__device__ float gaussianRayIntegral(float* &positions, float* &orientations,  float* &scales, float* &covs, int &gid, double t0,  double t1,const Ray& ray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    clock_t start = clock();
    
    Eigen::Vector3d omega(static_cast<double>(ray.direction.x),
                            static_cast<double>(ray.direction.y),
                            static_cast<double>(ray.direction.z));
    Eigen::Vector3d pos(positions[gid*3],positions[gid*3+1],positions[gid*3+2]);
    Eigen::Vector3d x(ray.source.x-positions[gid*3],ray.source.y-positions[gid*3+1],ray.source.z-positions[gid*3+2]);
    Eigen::Matrix3d Sigma;
    Sigma << static_cast<double>(covs[gid*9]), static_cast<double>(covs[gid*9+1]), static_cast<double>(covs[gid*9+2]),
         static_cast<double>(covs[gid*9+3]), static_cast<double>(covs[gid*9+4]), static_cast<double>(covs[gid*9+5]),
         static_cast<double>(covs[gid*9+6]), static_cast<double>(covs[gid*9+7]), static_cast<double>(covs[gid*9+8]);
    Eigen::Matrix3d orient;
    orient<<orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                                orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                                orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];

    

    const double c=10000000;
    //Eigen::Matrix3f SigmaInv = Sigma.inverse();
    Eigen::Matrix3d scaleInv=Eigen::Matrix3d::Zero();
    scaleInv(0,0)=1/(double)scales[gid*3];
    scaleInv(1,1)=1/(double)scales[gid*3+1];
    scaleInv(2,2)=1/(double)scales[gid*3+2];

    
    Eigen::Matrix3d SigmaInv = orient * scaleInv * scaleInv * (orient.transpose());
    clock_t end = clock();
    if(idx==500000)printf("Thread %d took %f ms\n", idx, (end - start) / (float)CLOCKS_PER_SEC * 1000.0);
    /*
    if(idx==500000){
    printf("scaleInv:\n");
    printMatrix3d(orient);
    printf("orient:\n");
    printMatrix3d(scaleInv);
    printf("SigmaInv:\n");
    printMatrix3d(SigmaInv);
    }
    */
    double A = omega.transpose() * SigmaInv * omega;
    double B = (omega.transpose() * SigmaInv * x + x.transpose() * SigmaInv * omega).sum();
    double C = x.transpose() * SigmaInv * x;

    
    //printf("A%lf B%lf C%lf\n",A,B,C);
    double expFactor = exp(B * B / (8 * A) - C / 2);
    
    if (isinf(expFactor)) {
        printf("inexp %f\n", B * B / (8 * A) - C / 2);
        return 0;
    }
    
    
    double tau0 = sqrt(A / 2) * t0 + B / (2 * sqrt(2 * A));
    double tau1 = sqrt(A / 2) * t1 + B / (2 * sqrt(2 * A));

    
    //if(idx==500000)printf("tau0%lf tau1%lf\n",tau0,tau1);
    
    
    double erfDiff = erf(tau1) - erf(tau0);

    double result = expFactor * sqrt(M_PI / (2 * A)) * erfDiff;
    
    
    if (isnan(erfDiff)) {
        printf("A %lf B %lf C %lf tau0 %f tau1 %f\n", A, B, C, tau0, tau1);
        printf("erfDiffnan\n");
    }
    if (isnan(sqrt(M_PI / (2 * A)))) {
        printf("sqrt\n");
    }
    if (isnan(expFactor)) {
        printf("expFactor\n");
    }
    
    //double detSigma = Sigma.determinant();
    //double constant = pow(2 * M_PI, 3.0 / 2.0);

    result = result * c;
    
    if (isnan(result)) {
        printf("result\n");
    }
    
    
    return result;
}

__device__ KDNode* front(KDNode*node,Ray& ray){
    float com=-1;
    switch(node->axis){
        case 0:com=ray.direction.x;break;
        case 1:com=ray.direction.y;break;
        case 2:com=ray.direction.z;break;
    }
    /*
    if(idx==12750){
        printf("com%d\n",com);
    }
    */
    if(com>0)return node->left;
    return node->right;
}

__device__ KDNode* back(KDNode*node,Ray& ray){
    float com=-1;
    switch(node->axis){
        case 0:com=ray.direction.x;break;
        case 1:com=ray.direction.y;break;
        case 2:com=ray.direction.z;break;
    }
    if(com<0)return node->left;
    return node->right;
}

__device__ void intersectBound(KDNode*node,Ray& ray,float& t0,float& t1){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    float invDirX = 1.0f / ray.direction.x; // 方向倒数，避免除零
    float tminX = (node->bbox.min.x - ray.source.x) * invDirX;
    float tmaxX = (node->bbox.max.x - ray.source.x) * invDirX;
    if (invDirX < 0.0f) swap(tminX, tmaxX); // 确保tminX < tmaxX
    //printf("tminX%f tmaxX%f\n",tminX,tmaxX);
    t0 = fmax(t0, tminX);
    t1 = fmin(t1, tmaxX);
    //if(idx==78*256)printf("t01 %f\n",t0);
    // y方向
    float invDirY = 1.0f / ray.direction.y;
    float tminY = (node->bbox.min.y - ray.source.y) * invDirY;
    float tmaxY = (node->bbox.max.y - ray.source.y) * invDirY;
    if (invDirY < 0.0f) swap(tminY, tmaxY);
    //printf("tminY%f tmaxY%f\n",tminY,tmaxY);
    t0 = fmax(t0, tminY);
    t1 = fmin(t1, tmaxY);
    //if(idx==78*256)printf("t02 %f\n",t0);
    // z方向
    float invDirZ = 1.0f / ray.direction.z;
    float tminZ = (node->bbox.min.z - ray.source.z) * invDirZ;
    float tmaxZ = (node->bbox.max.z - ray.source.z) * invDirZ;
    if (invDirZ < 0.0f) swap(tminZ, tmaxZ);
    //printf("tminZ%f tmaxZ%f\n",tminZ,tmaxZ);
    t0 = fmax(t0, tminZ);
    t1 = fmin(t1, tmaxZ);
    /*
    if(idx==78*256){
        printf("invDirZ%f t03 %f\n",invDirZ,t0);
    }
    */
}

__device__ int cntgs=0;

__device__ int sample(float* d_positions, float* d_orientations, float* d_scales, float* d_covs,float* magnitude,float* albedo, BoundingBox* d_bboxes, int N,
                      KDNode* root_node, Ray& ray, float t0, float t1, float u, float &cdf, int startid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //idx=12750;
    const int STACK_SIZE = 16;  // 根据你的需求调整栈大小
    struct StackFrame {
        KDNode* node;
        float t0;
        float t1;
        bool check_back;
    };

    StackFrame stack[STACK_SIZE];
    int stack_ptr = 0,max_len=0;

    stack[stack_ptr++] = {root_node, t0, t1, false};
    
    while (stack_ptr > 0) {
        max_len=max(max_len,stack_ptr);
        StackFrame frame = stack[--stack_ptr];
        KDNode* node = frame.node;
        float t0 = frame.t0;
        float t1 = frame.t1;
        
        if (!(node->is_leaf)) {
            // 处理非叶节点
            KDNode* frontNode = front(node, ray);
            KDNode* backNode = back(node, ray);
            float t0Front = t0, t1Front = t1, t0Back = t0, t1Back = t1;
            /*
            if(idx==12750){
                printf("node::minx%f maxx%f miny%f maxy%f minz%f maxz%f\n",node->bbox.min.x,node->bbox.max.x,node->bbox.min.y,node->bbox.max.y,node->bbox.min.z,node->bbox.max.z);
                printf("frontNode::minx%f maxx%f miny%f maxy%f minz%f maxz%f\n",frontNode->bbox.min.x,frontNode->bbox.max.x,frontNode->bbox.min.y,frontNode->bbox.max.y,frontNode->bbox.min.z,frontNode->bbox.max.z);
                printf("backNode::minx%f maxx%f miny%f maxy%f minz%f maxz%f\n",backNode->bbox.min.x,backNode->bbox.max.x,backNode->bbox.min.y,backNode->bbox.max.y,backNode->bbox.min.z,backNode->bbox.max.z);
                printf("\n");
            }
            */
            intersectBound(backNode, ray, t0Back, t1Back);
            intersectBound(frontNode, ray, t0Front, t1Front);
            /*
            if(idx==12750){
                printf("node::t0 %f t1 %f\n",t0,t1);
                printf("frontNode::t0 %f t1 %f\n",t0Front,t1Front);
                printf("backNode::t0 %f t1 %f\n",t0Back,t1Back);
                printf("\n");
            }
            */
            //if (t0Back < t1Back&&(backNode->gaussian_count)) {
            if (t0Back < t1Back) {
                stack[stack_ptr++] = {backNode, t0Back, t1Back, false};  // 后子节点入栈
            }

            

            //if (t0Front < t1Front&&(frontNode->gaussian_count)) {
            if (t0Front < t1Front) {
                // 先处理前子节点
                //stack[stack_ptr++] = {backNode, t0Back, t1Back, false};  // 后子节点入栈以备处理
                stack[stack_ptr++] = {frontNode, t0Front, t1Front, false};  // 前子节点入栈
            }

            /*
            if(idx==78*256)
            if(!(t0Front < t1Front||t0Back < t1Back)){
                printf("xfmin%f fmax%f bmin%f bmax%f\n",frontNode->bbox.min.x,frontNode->bbox.max.x,backNode->bbox.min.x,backNode->bbox.max.x);
                printf("yfmin%f fmax%f bmin%f bmax%f\n",frontNode->bbox.min.y,frontNode->bbox.max.y,backNode->bbox.min.y,backNode->bbox.max.y);
                printf("zfmin%f fmax%f bmin%f bmax%f\n",frontNode->bbox.min.z,frontNode->bbox.max.z,backNode->bbox.min.z,backNode->bbox.max.z);
                printf("t0Front%f t1Front%f t0Back%f t1Back%f\n",t0Front,t1Front,t0Back,t1Back);
                printf("\n");
                assert(0);
            }
            assert(t0Front < t1Front||t0Back < t1Back);
            */
        } else {
            // 处理叶节点
            int cnt = 0,pos=0;
            if(1)
            for (int i = 0; i < node->gaussian_count; i++) {
                int gsid = node->gaussian_indices[i];
                
                if (startid != gsid) {
                    //clock_t start = clock();
                    float integer = gaussianRayIntegral2(d_positions, d_orientations, d_scales, d_covs,magnitude, gsid,t0,t1, ray);
                    
                    //clock_t end = clock();
                    //if(idx==500000)printf("Thread %d took %f ms\n", idx, (end - start) / (float)CLOCKS_PER_SEC * 1000.0);

                    cdf += 0.5 * integer;
                    cnt= atomicAdd(&cntgs,1);
                    if (u < cdf ) {
                        if(idx==361048){
                            printf("integer%f\n",integer);
                        }
                        if(max_len>15)printf("max_len%d\n",max_len);
                        return gsid;  // 找到合适的高斯对象，返回其索引
                    }
                }
            }
            
            else
            for (int i = 0; i < 3; i++) {
                int gsid = 0;
                if(node->gaussian_count>0)gsid=node->gaussian_indices[0];
                cnt++;
                if (startid != gsid) {
                    //clock_t start = clock();
                    float integer = gaussianRayIntegral2(d_positions, d_orientations, d_scales, d_covs,magnitude, gsid,t0,t1,  ray);
                    //clock_t end = clock();
                    //if(idx==500000)printf("Thread %d took %f ms\n", idx, (end - start) / (float)CLOCKS_PER_SEC * 1000.0);
                    integer=0.02;
                    cdf += 0.5 * integer;
                    pos= atomicAdd(&cntgs,1); 
                    if (u < cdf  ) {
                        //if(max_len)printf("max_len%d\n",max_len);
                        return gsid;  // 找到合适的高斯对象，返回其索引
                    }
                }
            }
            
            
        }
    }
    return -1;  // 未找到任何满足条件的结果
}

__device__ Eigen::Vector3f transdir(Eigen::Vector3f dir,float* orientations,int gid){
    Eigen::Matrix3f orien;
    orien<< orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                            orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                            orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];
    Eigen::Vector3f dir_tan=orien.transpose()*dir;
    swap(dir_tan(2),dir_tan(1));
    return dir_tan;
}

__device__ Eigen::Vector3f invtransdir(Eigen::Vector3f dir_tan,float* orientations,int gid){
    Eigen::Matrix3f orien;
    orien<<orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                            orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                            orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];
    swap(dir_tan(2),dir_tan(1));
    Eigen::Vector3f dir=orien*dir_tan;
    return dir;
}

__device__ Eigen::Vector3f tan_to_world(float* d_positions, float* orientations,int gid,Eigen::Vector3f src){
    Eigen::Vector3f res(src(0)-d_positions[3*gid],src(1)-d_positions[3*gid+1],src(2)-d_positions[3*gid+2]);
    Eigen::Matrix3f orien;
    orien<<orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                            orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                            orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];
    res=orien.transpose()*res;
    return res;
}

__device__ void directionToSpherical(const float3* direction, float& theta, float& phi) {
    // 计算 theta（倾角），与 z 轴的夹角
    theta = asinf(direction->z);

    // 计算 phi（方位角），在 xy 平面上的角度
    phi = atan2f(direction->y, direction->x);
}

__device__ void sphericalToImageCoords(float theta, float phi,int& x,int& y) {
    // 将经度 theta 从 [-π, π] 映射到 [0, IMAGE_WIDTH-1]
    x = static_cast<int>((theta + (M_PI/2)) / (M_PI) * (4096 - 1));
    
    // 将纬度 phi 从 [-π/2, π/2] 映射到 [0, IMAGE_HEIGHT-1]
    y = static_cast<int>((phi + (M_PI)) / (2*M_PI) * (8192 - 1));
    
    // 保证坐标不越界
    x = max(0, min(4096 - 1, x));
    y = max(0, min(8192 - 1, y));

}

struct RayPayload {
    bool hasHit;              // 是否命中
    float3 hitPoint;          // 交点位置
    int triangleIndex;        // 三角形索引
};

extern "C" __global__ void __closesthit__radiance() {
    RayPayload* payload = reinterpret_cast<RayPayload*>(optixGetPayload_0());

    // 获取交点位置
    float3 dir=optixGetWorldRayDirection();
    float t=optixGetRayTmax();
    float3 source=optixGetWorldRayOrigin();
    float3 hitPoint=make_float3(source.x+dir.x*t,source.y+dir.y*t,source.z+dir.z*t);
    //float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    
    payload->hitPoint = hitPoint;

    // 获取三角形索引
    payload->triangleIndex = optixGetPrimitiveIndex();

    // 标记为命中
    payload->hasHit = true;
}

__device__ float3 shading(Ray &oray, float* d_positions, float* d_orientations, float* d_scales, float* d_covs,float* magnitude,float* albedo,
                           int N, OptixTraversableHandle& traversableHandle, int max_bound, curandState* state,
                          float* env_rgbs, float last_sca) {
    const float tMax = 1e16f; // 光线的最大追踪距离
    float cdf = 0.0f;         // 累积密度值
    float3 result = make_float3(1.0f, 1.0f, 1.0f); // 默认返回值

    RayPayload payload;
    payload.hasHit = false;   // 初始化为没有命中

    // 光线追踪循环
    while (cdf <= 1.0f) {
        // 初始化光线查询结果
        payload.hitPoint = make_float3(0.0f, 0.0f, 0.0f);
        payload.triangleIndex = -1; // 默认无效索引

        // 查询交点
        optixTrace(
            traversableHandle,             // BVH 句柄
            oray.source,               // 光线起点
            oray.direction,            // 光线方向
            0.0f,                    // tmin
            tMax,                    // tmax
            0.0f,                    // 光线时间（默认0）
            OptixVisibilityMask(1),  // 可见性掩码
            OPTIX_RAY_FLAG_NONE,     // 光线标志
            0,                       // SBT 索引
            1,                       // SBT 记录数
            0,                       // 光线类型索引
            payload                  // Payload 数据
        );

        if (!payload.hasHit) {
            // 没有更多交点，直接返回默认值
            return result;
        }

        // 获取交点位置和三角形索引
        float3 hitPoint = payload.hitPoint;
        int triangleIndex = payload.triangleIndex;
        int gsidx=triangleIndex/2;

        // 调用 integral 函数计算密度值
        float density=gaussianRayIntegral3(d_positions, d_orientations, d_scales, d_covs,magnitude,gsidx,hitPoint);


        // 累加到 cdf
        cdf += density;

        if (cdf > 1.0f) {
            // cdf 超过 1 时，返回 (0, 0, 0)
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        // 更新光线原点为当前交点位置，继续查找下一个交点
        oray.source = hitPoint;
    }

    // 如果 cdf 始终未超过 1，返回 (1, 1, 1)
    return result;

}

__device__ void pixel_to_ray_direction(
    const float* viewmatrix, // 4x4 view matrix in row-major order
    float tanfovx,           // Tangent of the horizontal field of view
    float tanfovy,           // Tangent of the vertical field of view
    int i,                   // Pixel row index
    int j,                   // Pixel column index
    int width,               // Image width
    int height,
    float3& ray_dir,
    float3& camera_pos
) {
    //printf("started_pixel_to_ray_direction\n");
    // 1. 计算像素在 NDC 坐标系中的归一化坐标
    float u = (j + 0.5f) / float(width);   // Normalize column index to [0, 1]
    float v = (i + 0.5f) / float(height);  // Normalize row index to [0, 1]

    // 2. 转换到视锥体坐标系（摄像机坐标系）
    float ray_cam[3];
    ray_cam[0] = (u - 0.5f) * 2.0f * tanfovx;  // x_cam
    ray_cam[1] = (v-  0.5f) * 2.0f * tanfovy;  // y_cam
    ray_cam[2] = -1.0f;                        // z_cam (camera looks along -z)

    // 3. 使用视图矩阵将光线从摄像机坐标系转换到世界坐标系
    float R[9] = {viewmatrix[0], viewmatrix[1], viewmatrix[2],  // Row 1
                  viewmatrix[4], viewmatrix[5], viewmatrix[6],  // Row 2
                  viewmatrix[8], viewmatrix[9], viewmatrix[10]};// Row 3
    float ray_world[3];
    float RT[9];
    matrixTranspose1(R,RT);
    matrixVectorMultiply1(RT, ray_cam, ray_world);

    // 4. 归一化结果
    float length = sqrtf(ray_world[0] * ray_world[0] +
                         ray_world[1] * ray_world[1] +
                         ray_world[2] * ray_world[2]);
    ray_dir=make_float3(ray_world[0] / length,ray_world[1] / length,ray_world[2] / length);

    float translation[3] = {viewmatrix[3], viewmatrix[7], viewmatrix[11]};

    float camera_p[3];
    // 计算摄像机位置 T = -R^T * translation
    for (int i = 0; i < 3; ++i) {
        camera_p[i] = -(R[0 + i] * translation[0] +
                          R[3 + i] * translation[1] +
                          R[6 + i] * translation[2]);
    }
    camera_pos=make_float3(camera_p[0],camera_p[1],camera_p[2]);
    //printf("end_pixel_to_ray_direction\n");
}

__global__ void renderPixel(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,float* magnitude,float* albedo,int N,OptixTraversableHandle& traversableHandle,int max_depth,
int max_bound,int res,int focal,curandState *state,float3* rgbs,float* env_rgbs,float* view_mat,float tanfovx,float tanfovy){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //idx=12750;
    if(idx==12750)printf("start_kernel_render\n");

    int i=idx/res;
    int j=idx%res;
    
    //float3 pix_pos=make_float3(1,(float)i / res+(float)1/(2*res)  , (float)j / res+(float)1/(2*res)  );
    //float3 source=make_float3(1+focal, 0.5,0.5);
    float3 pix_pos=make_float3(-3,(float)i / res+(float)1/(2*res)  , (float)j / res+(float)1/(2*res)  );
    float3 source=make_float3(-focal-3, 0.5,0.5);
    //float3 pix_pos=make_float3((float)i / res+(float)1/(2*res)  , (float)j / res+(float)1/(2*res),0 );
    //float3 source=make_float3(0.5,0.5,-focal);

    float3 direction;
    direction.x=pix_pos.x-source.x;direction.y=pix_pos.y-source.y;direction.z=pix_pos.z-source.z;
    float inv_length = rnorm3df(direction.x, direction.y, direction.z);
    direction.x*=inv_length;
    direction.y*=inv_length;
    direction.z*=inv_length;
    
    //float3 direction,source;
    //pixel_to_ray_direction(view_mat,tanfovx,tanfovy,i,j,res,res,direction,source);
    Ray ray;
    ray.source=source;
    ray.direction=direction;
    if(idx==0){
        printf("dirx%f diry%f dirz%f\n",direction.x,direction.y,direction.z);
        printf("sorx%f sory%f sorz%f\n",source.x,source.y,source.z);
    }
    //printf("sourx%f soury%f sourz%f dirx%f diry%f dirz%f\n",ray.source.x,ray.source.y,ray.source.z,ray.direction.x,ray.direction.y,ray.direction.z);
    rgbs[idx]=shading(ray,d_positions, d_orientations, d_scales,d_covs,magnitude,albedo,N,traversableHandle,max_bound,state,env_rgbs,0);
    //printf("rgb%f\n",rgbs[idx].x);
    //if(rgbs[idx].x!=0)printf("r%f g%f b%f\n",rgbs[idx].x,rgbs[idx].y,rgbs[idx].z);
}

void renderScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,float* magnitude,float* albedo,int N,OptixTraversableHandle& traversableHandle,int max_depth,int max_bound,
int res,int focal,float* rgbs_sc,float* env_rgbs,float* view_mat,float tanfovx,float tanfovy){
    size_t newSize=2048;
    cudaDeviceSetLimit(cudaLimitStackSize, newSize);
    //printf("stackSize%d\n",stackSize);
    int threadsPerBlock = 256;
    int blocksPerGrid = (res*res) / threadsPerBlock;
    curandState *d_state;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_state, res*res * sizeof(curandState)));
    float3* rgbs=(float3*)rgbs_sc;
    //printf("rgb_sc%d rgbs%d\n",rgb_sc,rgbs);
    //CHECK_CUDA_ERROR(cudaMalloc((void **)&rgbs, res*res * sizeof(float3)));
    //rgbs_sc=(float*)rgbs;
    setupKernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, time(0));
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    clock_t st=clock();
    renderPixel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_orientations, d_scales,d_covs,magnitude,albedo,N,traversableHandle,15,1,res,focal,d_state,rgbs,env_rgbs,view_mat,tanfovx,tanfovy);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error renderPixel: %s\n", cudaGetErrorString(err));
    }
    //printf("finished_render\n");
    clock_t ed=clock();
    printf("time%lf CLOCKS_PER_SEC%d\n",(double)(ed-st),CLOCKS_PER_SEC);
    int cntgs_h;
    CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&cntgs_h, cntgs, sizeof(int), 0, cudaMemcpyDeviceToHost));
    printf("cntgs%d\n",cntgs_h);
    printf("finished_render\n");

}

