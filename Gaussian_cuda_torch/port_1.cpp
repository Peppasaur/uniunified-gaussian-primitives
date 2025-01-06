#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include"gaussian_bbox.h"
int render_c(torch::Tensor x){
    return 1;
}
/*
float* render_c1(float* position,float* scale,float* orientation,int magnitude,float* &rgbs,float* env_rgbs){
    float* d_env_rgbs;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_env_rgbs, 8192*4096 *3* sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_env_rgbs, env_rgbs, 8192*4096 *3* sizeof(float),cudaMemcpyHostToDevice));
    CreateScene(d_positions, d_orientations,  d_scales,d_covs,gaussians.size(),1024,d_env_rgbs);
    return rgbs;
}
*/
PYBIND11_MODULE(cuda_renderer, m) {
    m.def("render_c", &render_c, "scene rendering");
}