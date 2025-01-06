#ifndef RENDER_C
#define RENDER_C
#include"gaussian_bbox.h"
#include <cuda_runtime.h>
    void renderScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,BoundingBox* d_bboxes,int N,KDNode* d_nodes,int max_depth,
    int res,int focal);
#endif