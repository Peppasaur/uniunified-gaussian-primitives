#include <iostream>
#include<cmath>
#include <cstring> 
#include <cuda_runtime.h>
#include"gaussian_bbox.h"
#include"time.h"
#include"render_c.h"
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

#define CHECK(call)                                                             \
do                                                                              \
{                                                                               \
    const cudaError_t error_code = call;                                        \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
        printf("CUDA Error:\n");                                                \
        printf("     File:      %s\n", __FILE__);                               \
        printf("     Line       %d:\n", __LINE__);                              \
        printf("     Error code:%d\n", error_code);                             \
        printf("     Error text:%s\n", cudaGetErrorString(error_code));         \
        exit(1);                                                                \
    }                                                                           \
}while(0) 

__device__  int gs_bound=0;
__device__ bool intersects(const BoundingBox& a, const BoundingBox& b){
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
               (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
               (a.min.z <= b.max.z && a.max.z >= b.min.z);
}
// CUDA 核函数，计算每个 Gaussian 的包围盒
__global__ void computeBoundingBoxKernel(
    float* d_positions,      // 高斯的位置数组 [3 * N]（已在 GPU 上）
    float* d_orientations,   // 高斯的旋转矩阵数组 [9 * N]（已在 GPU 上）
    float* d_scales,         // 高斯的尺度数组 [3 * N]（已在 GPU 上）
    BoundingBox* d_bboxes,   // 输出的包围盒数组（在 GPU 上）
    int N                    // Gaussian 数量
) {
    //printf("111\n");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 每个 Gaussian 对应的下标
        int pos_offset = idx * 3;
        int orient_offset = idx * 9;
        
        // 获取 Gaussian 的位置、旋转矩阵、尺度
        float3 position = make_float3(d_positions[pos_offset], d_positions[pos_offset + 1], d_positions[pos_offset + 2]);
        float3 scale = make_float3(d_scales[pos_offset], d_scales[pos_offset + 1], d_scales[pos_offset + 2]);

        // 计算包围盒，这里假设 orientation 不影响计算，仅考虑 position 和 scale
        float sca=fmax(d_scales[pos_offset],fmax(d_scales[pos_offset+1],d_scales[pos_offset+2]));
        //printf("N%d\n",N);
        
        float mat=0.001;
        float exp_bound=0.01;
        float bound_2 = -2.0f * logf(exp_bound) * powf(sca, 2.0f);
        float bound = sqrtf(bound_2);
        //bound=sca;
        //printf("bound%f\n",bound);
        
        d_bboxes[idx].min = make_float3(d_positions[pos_offset] - bound, d_positions[pos_offset+1] - bound, d_positions[pos_offset+2] - bound);
        d_bboxes[idx].max = make_float3(d_positions[pos_offset] + bound, d_positions[pos_offset+1] + bound, d_positions[pos_offset+2] + bound);
        /*
        float3 half_scale = make_float3(scale.x / 2, scale.y / 2, scale.z / 2);
        d_bboxes[idx].min = make_float3(position.x - half_scale.x, position.y - half_scale.y, position.z - half_scale.z);
        d_bboxes[idx].max = make_float3(position.x + half_scale.x, position.y + half_scale.y, position.z + half_scale.z);
        */
    
    }
}

__global__ void levelProcess(KDNode* nodes,  int depth,int max_depth,int* all_indices) {
    //printf("started_level\n");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int range=pow(max_depth-1,2)/pow(depth-1,2);
    int range=(1<<(max_depth-depth));
    int node_off=idx/range;
    //int node_base=pow(depth-1,2)-1;
    int node_base=(1<<(depth-1))-1;
    int node_id=node_base+node_off;
    int rk=idx%range;
    
    KDNode* lnode=&nodes[node_id*2+1];
    KDNode* rnode=&nodes[node_id*2+2];
    KDNode* node=&nodes[node_id];

    if(rk==0){
        int axis = (depth+1) % 3; // 0 for x, 1 for y, 2 for z
        float splitValue;

        if (axis == 0) { // x 轴
            splitValue = 0.5f * ((node->bbox).min.x + (node->bbox).max.x);
        } else if (axis == 1) { // y 轴
            splitValue = 0.5f * ((node->bbox).min.y + (node->bbox).max.y);
        } else { // z 轴
            splitValue = 0.5f * ((node->bbox).min.z + (node->bbox).max.z);
        }
        node->is_leaf=0;
        node->axis=axis;
        node->left=lnode;
        node->right=rnode;
        // 定义子节点的包围盒
        lnode->bbox=node->bbox;
        rnode->bbox=node->bbox;
        if (axis == 0) {
            (lnode->bbox).max.x = splitValue;
            (rnode->bbox).min.x = splitValue;
        } else if (axis == 1) {
            (lnode->bbox).max.y = splitValue;
            (rnode->bbox).min.y = splitValue;
        } else { // axis == 2
            (lnode->bbox).max.z = splitValue;
            (rnode->bbox).min.z = splitValue;
        }
        int lpos = atomicAdd(&gs_bound,node->gaussian_count); 
        lnode->gaussian_indices=all_indices+lpos;
        int rpos = atomicAdd(&gs_bound,node->gaussian_count); 
        rnode->gaussian_indices=all_indices+rpos;
        //if(idx==0)printf("node->gaussian_count%d gs_bound%d\n",node->gaussian_count,gs_bound);
        lnode->gaussian_count=0;
        rnode->gaussian_count=0;
        //cudaMalloc((void**)&lnode->gaussian_indices, node->gaussian_count * sizeof(int));
        //cudaMalloc((void**)&rnode->gaussian_indices, node->gaussian_count * sizeof(int));
        if(depth==max_depth-1){
            lnode->is_leaf=1;
            rnode->is_leaf=1;
        }
    }   
}

__global__ void buildKDTreeKernel(KDNode* nodes, const BoundingBox* bboxes, int depth,int max_depth,int* all_indices
) {
    //printf("started_build\n");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int range=pow(max_depth-1,2)/pow(depth-1,2);
    int range=(1<<(max_depth-depth));
    int node_off=idx/range;
    //int node_base=pow(depth-1,2)-1;
    int node_base=(1<<(depth-1))-1;
    int node_id=node_base+node_off;
    int rk=idx%range;
    //if(rk==0)

    KDNode* lnode=&nodes[node_id*2+1];
    KDNode* rnode=&nodes[node_id*2+2];
    KDNode* node=&nodes[node_id];
    int thread_sz=(node->gaussian_count+range-1)/range;
    thread_sz=max(thread_sz,1);
    
    //printf("node->gaussian_count%d\n",node->gaussian_count);
    for (int i = rk*thread_sz; i < (rk+1)*thread_sz; i++) {

        if(i>=node->gaussian_count)break;
        int gs_idx = node->gaussian_indices[i];
        BoundingBox bidx=bboxes[gs_idx];
        // 这里需要实现包围盒的相交测试
        //printf("gs_boundloop%d\n",gs_bound);
        if (intersects(lnode->bbox, bboxes[gs_idx])) {
            // 如果与父节点相交，则将高斯对象分配到左子节点或右子节点
            int pos = atomicAdd(&lnode->gaussian_count,1); 
            lnode->gaussian_indices[pos]=gs_idx;
        }
        if (intersects(rnode->bbox, bboxes[gs_idx])) {
            // 如果与父节点相交，则将高斯对象分配到左子节点或右子节点
            int pos = atomicAdd(&rnode->gaussian_count, 1); 
            rnode->gaussian_indices[pos]=gs_idx;
        }
        //printf("endloop\n");
        /*
        if (1) {
            // 如果与父节点相交，则将高斯对象分配到左子节点或右子节点
            int pos = atomicAdd(&lnode->gaussian_count,1); 
            lnode->gaussian_indices[pos]=gs_idx;
        }
        if (1) {
            // 如果与父节点相交，则将高斯对象分配到左子节点或右子节点
            int pos = atomicAdd(&rnode->gaussian_count, 1); 
            rnode->gaussian_indices[pos]=gs_idx;
        }
        */
    }
    //printf("end_build\n");
    __syncthreads();
    if(rk==0){

        //if(lnode->is_leaf&&lnode->gaussian_count>5)printf("lnode->gaussian_count%d\n",lnode->gaussian_count);
        //if(rnode->is_leaf&&rnode->gaussian_count>5)printf("rnode->gaussian_count%d\n",rnode->gaussian_count);
    }
    /*
    if(rk==0){
        int* old_indices_l=lnode->gaussian_indices,* old_indices_r=rnode->gaussian_indices;
        cudaMalloc((void**)&lnode->gaussian_indices, lnode->gaussian_count * sizeof(int));
        cudaMalloc((void**)&rnode->gaussian_indices, rnode->gaussian_count * sizeof(int));
        for(int)
        cudaFree(old_indices_l);
        cudaFree(old_indices_r);
    }
    */
    // 根据高斯对象的数量分配子节点
    //node->gaussian_count = left_count; // 假设更新子节点的高斯数量
    //node->gaussian_indices = new int[left_count]; // 申请内存，实际实现时需要分配合适的内存

}

__global__ void test(){
    printf("gs_bound%d\n",gs_bound);
}
// CPU 函数，调用 CUDA 核函数计算包围盒
void computeBoundingBoxes(float* d_positions, float* d_orientations, float* d_scales,BoundingBox* d_bboxes,int N) {
    // 计算 CUDA 核函数所需的线程块和网格大小
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 调用核函数，直接使用已在 GPU 上的数组
    computeBoundingBoxKernel<<<gridSize, blockSize>>>(d_positions, d_orientations, d_scales, d_bboxes, N);
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error bounding: %s\n", cudaGetErrorString(err));
    }
}

void initKDTree(KDNode* root,int N,int* all_indices,float* space){
    //printf("startinit\n");
    float3 bmin{space[0],space[1],space[2]};
    float3 bmax{space[3],space[4],space[5]};
    BoundingBox bbox{bmin,bmax};
    KDNode temp;
    temp.bbox=bbox;
    temp.gaussian_count=N;
    temp.gaussian_indices=all_indices;
    CHECK_CUDA_ERROR(cudaMemcpy(root, &temp, sizeof(KDNode),cudaMemcpyHostToDevice));

    temp.gaussian_indices=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        temp.gaussian_indices[i]=i;
    }
    //printf("startmal3 N%d\n",N);
    CHECK_CUDA_ERROR(cudaMemcpy(all_indices, temp.gaussian_indices, N*sizeof(int),cudaMemcpyHostToDevice));
    free(temp.gaussian_indices);
}

void buildKDTree(const BoundingBox* bboxes, int N, int max_depth,KDNode*& d_nodes,int*& all_indices,float* space) {

    int h_data = 5; 
    int* d_data;
    CHECK(cudaMalloc((void**)&d_data, 3*sizeof(int)));
    CHECK(cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));

    int node_count = (1<<max_depth)-1; // 假设节点数量等于包围盒数量
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nodes, node_count * sizeof(KDNode)));
    //int* all_indices;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&all_indices, max_depth*4*N * sizeof(int)));
    printf("afterMalloc2");
    initKDTree(d_nodes,N,all_indices,space);
    printf("afterMalloc3\n");
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gs_bound, &N, sizeof(int)););
    int threadsPerBlock = 1;
    int blocksPerGrid = (1<<(max_depth-1)) / threadsPerBlock;
    // 启动核函数
    printf("startingCore\n");
    clock_t st=clock();
    for(int i=1;i<=max_depth-1;i++){
        levelProcess<<<blocksPerGrid, threadsPerBlock>>>(d_nodes,i,max_depth,all_indices);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error levelProcess: %s\n", cudaGetErrorString(err));
        }
        //test<<<1,1>>>();
        buildKDTreeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, bboxes, i, max_depth,all_indices);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error buildkdtree: %s\n", cudaGetErrorString(err));
        }
        
    }
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    clock_t ed=clock();
    printf("time%lf CLOCKS_PER_SEC%d\n",(double)(ed-st),CLOCKS_PER_SEC);
    printf("finished\n");
    return;
}



void CreateScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,float* magnitude,float* albedo,int N,float* rgbs,int res,float* env_rgbs,float* view_mat,float* space,float tanfovx,float tanfovy){
    BoundingBox* d_bboxes;
    CHECK(cudaMalloc((void**)&d_bboxes, N*sizeof(BoundingBox)));
    computeBoundingBoxes(d_positions, d_orientations, d_scales,d_bboxes,N);
    //return;
    KDNode* d_nodes;
    int* all_indices;
    buildKDTree(d_bboxes,N,12,d_nodes,all_indices,space);
    
    renderScene(d_positions, d_orientations, d_scales,d_covs,magnitude,albedo,d_bboxes,N,d_nodes,15,4,res,1,rgbs,env_rgbs,view_mat,tanfovx,tanfovy);
    CHECK(cudaFree(d_nodes));
    CHECK(cudaFree(all_indices));
    CHECK(cudaFree(d_bboxes));
}

int rua(){
    return 2;
}
