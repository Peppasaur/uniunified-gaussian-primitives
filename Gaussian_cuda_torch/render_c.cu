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

__device__ Eigen::Vector3d intersectRayWithPlane_d(Ray &ray,float* d_positions,int gid,
                                              const Eigen::Vector3d& planeNormal) {
    Eigen::Vector3d rayOrigin(ray.source.x,ray.source.y,ray.source.z);
    Eigen::Vector3d rayDir(ray.direction.x,ray.direction.y,ray.direction.z);
    Eigen::Vector3d pointOnPlane(d_positions[3*gid],d_positions[3*gid+1],d_positions[3*gid+2]);
    double denominator = planeNormal.dot(rayDir);
    
    // 如果点积为0，说明射线与平面平行或在平面上
    /*
    if (std::abs(denominator) < 1e-6) {
        return Eigen::Vec; // 没有交点
    }
    */

    // 计算参数 t
    double t = planeNormal.dot(pointOnPlane - rayOrigin) / denominator;
    
    // 如果 t < 0，说明交点在射线的反方向上
    /*
    if (t < 0) {
        return NULL; // 射线与平面没有交点（射线背面）
    }
    */

    // 计算交点 P = O + t * D
    Eigen::Vector3d intersectionPoint = rayOrigin + t * rayDir;
    return intersectionPoint;
}

__device__ float gaussianRayIntegral2(float* &positions, float* &orientations, float* &scales, float* &covs, int &gid ,float& t0, float& t1, Ray& ray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Eigen::Vector3f norm(orientations[9*gid+2],orientations[9*gid+5],orientations[9*gid+8]);
    Eigen::Vector3f ray_dir(ray.direction.x, ray.direction.y, ray.direction.z);
    float dot = norm.transpose() * (-ray_dir);
    if(dot<0){
        return 0;
        //printf("-dot\n");
    }
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
    return result*0.01;
}

__device__ float gaussianRayIntegral3(float* &positions, float* &orientations, float* &scales, float* &covs, int &gid ,float& t0, float& t1, Ray& ray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Eigen::Vector3d norm(orientations[9*gid+2],orientations[9*gid+5],orientations[9*gid+8]);
    Eigen::Matrix3d orient;
    orient<<orientations[gid*9],orientations[gid*9+1],orientations[gid*9+2],
                                orientations[gid*9+3],orientations[gid*9+4],orientations[gid*9+5],
                                orientations[gid*9+6],orientations[gid*9+7],orientations[gid*9+8];
    Eigen::Vector3d pos(positions[gid*3],positions[gid*3+1],positions[gid*3+2]);
    Eigen::Vector3d point=intersectRayWithPlane_d(ray,positions,gid,norm);
    point=point-pos;
    Eigen::Vector3d loc_point=orient.transpose()*point;
    double scalex=scales[3*gid],scaley=scales[3*gid+1];
    double normalization_factor = 1.0f / (2.0f * M_PI * scalex * scaley);
    double exponent_x = - (loc_point(0) * loc_point(0)) / (2.0f * scalex*scalex);
    double exponent_y = - (loc_point(1) * loc_point(1)) / (2.0f * scaley*scaley);

    double result =normalization_factor * expf(exponent_x + exponent_y);
    if (isnan(result)) {
        printf("nanresult\n");
    }
    if (isinf(result)) {
        printf("infresult\n");
    }
    //if(result>0.0000001)printf("result %f\n",result);
    return result*0.001;
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
    if(idx==524750){
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

__device__ int sample(float* d_positions, float* d_orientations, float* d_scales, float* d_covs, BoundingBox* d_bboxes, int N,
                      KDNode* root_node, Ray& ray, float t0, float t1, float u, float &cdf, int startid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
            if(idx==524750){
                printf("node::minx%f maxx%f miny%f maxy%f minz%f maxz%f\n",node->bbox.min.x,node->bbox.max.x,node->bbox.min.y,node->bbox.max.y,node->bbox.min.z,node->bbox.max.z);
                printf("frontNode::minx%f maxx%f miny%f maxy%f minz%f maxz%f\n",frontNode->bbox.min.x,frontNode->bbox.max.x,frontNode->bbox.min.y,frontNode->bbox.max.y,frontNode->bbox.min.z,frontNode->bbox.max.z);
                printf("backNode::minx%f maxx%f miny%f maxy%f minz%f maxz%f\n",backNode->bbox.min.x,backNode->bbox.max.x,backNode->bbox.min.y,backNode->bbox.max.y,backNode->bbox.min.z,backNode->bbox.max.z);
                printf("\n");
            }
            */
            intersectBound(backNode, ray, t0Back, t1Back);
            intersectBound(frontNode, ray, t0Front, t1Front);
            /*
            if(idx==524750){
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
                    float integer = gaussianRayIntegral2(d_positions, d_orientations, d_scales, d_covs, gsid,t0,t1, ray);
                    
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
                    float integer = gaussianRayIntegral2(d_positions, d_orientations, d_scales, d_covs, gsid,t0,t1,  ray);
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

__device__ float3 shading(Ray &ray, float* d_positions, float* d_orientations, float* d_scales, float* d_covs,
                          BoundingBox* d_bboxes, int N, KDNode* d_nodes, int max_bound, curandState* state,
                          float* env_rgbs, float last_sca) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 dark = make_float3(0, 0, 0);
    float3 ambient = make_float3(100.0f / 255, 100.0f / 255, 100.0f / 255);
    
    int bound = 0,start_gs=-1;

    while (bound < max_bound) {
        float t0 = 5*last_sca, t1 = 100, cdf = 0;
        intersectBound(d_nodes, ray, t0, t1);

        float u = curand_uniform(&state[idx]);;  // 固定随机数
        int res = sample(d_positions, d_orientations, d_scales, d_covs, d_bboxes, N, d_nodes, ray, t0, t1, u, cdf, start_gs);

        if (res != -1) {
            // 获取 ray_dir 和 norm
            Eigen::Vector3f ray_dir(ray.direction.x, ray.direction.y, ray.direction.z);
            Eigen::Vector3f norm(d_orientations[9 * res + 2], d_orientations[9 * res + 5], d_orientations[9 * res + 8]);
            float dot = norm.transpose() * (-ray_dir);

            if (dot < 0) {
                // 如果需要，可以在此反转 norm 的方向
            }

            // 计算反射方向
            Eigen::Vector3f dir_tan = transdir(-ray_dir, d_orientations, res);
            float3 v = make_float3(dir_tan(0), dir_tan(1), dir_tan(2));
            Eigen::Vector3f dir(-v.x, v.y, -v.z);
            dir = invtransdir(dir, d_orientations, res);

            // 计算新的光线起点
            Eigen::Vector3f src = intersectRayWithPlane(ray, d_positions, res, norm);

            // 更新 `ray` 为新的反射光线
            ray.source.x = src(0); ray.source.y = src(1); ray.source.z = src(2);
            ray.direction.x = dir(0); ray.direction.y = dir(1); ray.direction.z = dir(2);

            // 更新上一次的尺度
            last_sca = d_scales[3 * res];

            // 增加反射次数计数
            bound++;

            start_gs=res;
        } else {
            
            // 如果没有交点，计算环境光颜色并返回
            float theta = 0, phi = 0;
            int x = 0, y = 0;

            directionToSpherical(&ray.direction, theta, phi);
            sphericalToImageCoords(theta, phi, x, y);
            int env_pix = x * 8192 + y;
            float3 env_rgb = make_float3(env_rgbs[3 * env_pix], env_rgbs[3 * env_pix + 1], env_rgbs[3 * env_pix + 2]);

            return env_rgb;
        }
    }
    if(idx==361048){
        printf("captured\n");
    }
    return dark;
}


__global__ void renderPixel(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,BoundingBox* d_bboxes,int N,KDNode* d_nodes,int max_depth,
int max_bound,int res,int focal,curandState *state,float3* rgbs,float* env_rgbs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if((idx/512)%2==1)return;
    /*
    Ray ray1;
    ray1.source=make_float3(2,0.5,0.5);
    float3 pix=make_float3(1,0.6,0.6);
    float3 direction1=make_float3(-1,0.1,0.1);
    float inv_length1 = rnorm3df(direction1.x, direction1.y, direction1.z);
    direction1.x*=inv_length1;
    direction1.y*=inv_length1;
    direction1.z*=inv_length1;
    ray1.direction=direction1;
    float integer=0;
    
    double t0=0,t1=1;
    
    int gid=0;
    //float mat1[9]={12425.00003,1245.00003,75856.00003,75856.00003,5426.00003,679.00003,3546.00003,647.00003,2356.00003};
    //float mat2[9]={5346.00003,768.00003,356.00003,785567.00003,4537.00003,567.00003,25.00003,568.00003,356.00003};
    //float mat3[9]={0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};
    //for(int j=0;j<9;j++)mat1[j]=d_orientations[(gid+j)%N];
    //for(int j=0;j<9;j++)mat2[j]=d_orientations[(gid+j)%N];
    for(int i=0;i<50;i++){
    float integer = gaussianRayIntegral2(d_positions, d_orientations, d_scales, d_covs, gid, t0, t1, ray1);
    
    if(integer>10000000000)printf("miracle\n");
    
    }
    return;
    */


    
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i=idx/res;
    int j=idx%res;
    //float3 pix_pos=make_float3(1,(float)i / res+(float)1/(2*res)  , (float)j / res+(float)1/(2*res)  );
    //float3 source=make_float3(1+focal, 0.5,0.5);
    float3 pix_pos=make_float3(0,(float)i / res+(float)1/(2*res)  , (float)j / res+(float)1/(2*res)  );
    float3 source=make_float3(-focal, 0.5,0.5);
    //float3 pix_pos=make_float3((float)i / res+(float)1/(2*res)  , (float)j / res+(float)1/(2*res),0 );
    //float3 source=make_float3(0.5,0.5,-focal);

    float3 direction;
    direction.x=pix_pos.x-source.x;direction.y=pix_pos.y-source.y;direction.z=pix_pos.z-source.z;
    float inv_length = rnorm3df(direction.x, direction.y, direction.z);
    direction.x*=inv_length;
    direction.y*=inv_length;
    direction.z*=inv_length;
    Ray ray;
    ray.source=source;
    ray.direction=direction;
    //printf("sourx%f soury%f sourz%f dirx%f diry%f dirz%f\n",ray.source.x,ray.source.y,ray.source.z,ray.direction.x,ray.direction.y,ray.direction.z);
    rgbs[idx]=shading(ray,d_positions, d_orientations, d_scales,d_covs,d_bboxes,N,d_nodes,max_bound,state,env_rgbs,0);
    //printf("rgb%f\n",rgbs[idx].x);
    //if(rgbs[idx].x!=0)printf("r%f g%f b%f\n",rgbs[idx].x,rgbs[idx].y,rgbs[idx].z);
}

void renderScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,BoundingBox* d_bboxes,int N,KDNode* d_nodes,int max_depth,int max_bound,
int res,int focal,float* rgbs_sc,float* env_rgbs){
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

    renderPixel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_orientations, d_scales,d_covs,d_bboxes,N,d_nodes,15,3,res,focal,d_state,rgbs,env_rgbs);
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error renderPixel: %s\n", cudaGetErrorString(err));
    }
    clock_t ed=clock();
    printf("time%lf CLOCKS_PER_SEC%d\n",(double)(ed-st),CLOCKS_PER_SEC);
    int cntgs_h;
    CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&cntgs_h, cntgs, sizeof(int), 0, cudaMemcpyDeviceToHost));
    printf("cntgs%d\n",cntgs_h);
    printf("finished_render\n");

}

