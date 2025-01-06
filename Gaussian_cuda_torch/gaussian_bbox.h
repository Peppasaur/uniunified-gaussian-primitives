#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <cuda_runtime.h>

static inline __device__ void swap(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}

// 包围盒的数据结构定义
struct BoundingBox {
    float3 min;  // 包围盒的最小值 (x_min, y_min, z_min)
    float3 max;  // 包围盒的最大值 (x_max, y_max, z_max)
};

struct Ray {
    float3 source,direction;
};

struct KDNode {
    int axis;            // 分割轴 (0: x, 1: y, 2: z)
    float split;     // 分割位置
    BoundingBox bbox;    // 节点对应的包围盒
    int* gaussian_indices; // 指向高斯对象索引的指针
    int st,ed;
    unsigned int gaussian_count;   // 存储高斯对象的数量
    bool is_leaf;         // 是否是叶子节点
    KDNode* left,*right;
    __device__ void intersectBound(Ray& ray,float& t0,float& t1);
    __device__ KDNode* front(Ray& ray);
    __device__ KDNode* back(Ray& ray);
};

// KD 树结构体定义
struct KDTree {
    KDNode* root;       // KD 树节点数组
    int max_depth;       // 最大深度
    int gs_bound;
};



void computeBoundingBoxes(
    float* d_positions,      // 高斯的位置数组 [3 * N]（已在 GPU 上）
    float* d_orientations,   // 高斯的旋转矩阵数组 [9 * N]（已在 GPU 上）
    float* d_scales,         // 高斯的尺度数组 [3 * N]（已在 GPU 上）
    BoundingBox* d_bboxes,   // 输出的包围盒数组（在 GPU 上）
    int N                    // Gaussian 数量
);



void buildKDTree(const BoundingBox* bboxes, int n, int max_depth);
void CreateScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,int N,float* rgbs,int res,float* env_rgbs);
void renderScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,BoundingBox* d_bboxes,int N,KDNode* d_nodes,int max_depth,int max_bound,
int res,int focal,float* rgbs,float* env_rgbs);
int rua();



#endif // BOUNDINGBOX_H
