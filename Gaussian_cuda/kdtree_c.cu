#include <curand_kernel.h>
#include"gaussian_bbox.h"



__device__ KDNode* KDNode::front(Ray& ray){
    float com=0;
    switch(this->axis){
        case 0:com=ray.direction.x;
        case 1:com=ray.direction.y;
        case 2:com=ray.direction.z;
    }
    if(com>0)return this->left;
    return this->right;
}

__device__ KDNode* KDNode::back(Ray& ray){
    float com=0;
    switch(this->axis){
        case 0:com=ray.direction.x;
        case 1:com=ray.direction.y;
        case 2:com=ray.direction.z;
    }
    if(com<0)return this->left;
    return this->right;
}

__device__ void KDNode::intersectBound(Ray& ray,float& t0,float& t1){
            // x方向
    float invDirX = 1.0f / ray.direction.x; // 方向倒数，避免除零
    float tminX = (bbox.min.x - ray.source.x) * invDirX;
    float tmaxX = (bbox.max.x - ray.source.x) * invDirX;
    if (invDirX < 0.0f) swap(tminX, tmaxX); // 确保tminX < tmaxX
    //printf("tminX%f tmaxX%f\n",tminX,tmaxX);
    t0 = fmax(t0, tminX);
    t1 = fmin(t1, tmaxX);
    
    // y方向
    float invDirY = 1.0f / ray.direction.y;
    float tminY = (bbox.min.y - ray.source.y) * invDirY;
    float tmaxY = (bbox.max.y - ray.source.y) * invDirY;
    if (invDirY < 0.0f) swap(tminY, tmaxY);
    //printf("tminY%f tmaxY%f\n",tminY,tmaxY);
    t0 = fmax(t0, tminY);
    t1 = fmin(t1, tmaxY);
    
    // z方向
    float invDirZ = 1.0f / ray.direction.z;
    float tminZ = (bbox.min.z - ray.source.z) * invDirZ;
    float tmaxZ = (bbox.max.z - ray.source.z) * invDirZ;
    if (invDirZ < 0.0f) swap(tminZ, tmaxZ);
    //printf("tminZ%f tmaxZ%f\n",tminZ,tmaxZ);
    t0 = fmax(t0, tminZ);
    t1 = fmin(t1, tmaxZ);

}
