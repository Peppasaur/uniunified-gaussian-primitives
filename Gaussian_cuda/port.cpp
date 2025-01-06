#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include"gaussian_bbox.h"

void render_c(torch::Tensor& position,torch::Tensor& scale,torch::Tensor& orientation,torch::Tensor& cov,torch::Tensor& magnitude,
torch::Tensor& albedo,torch::Tensor& env_rgbs,torch::Tensor& view_mat,torch::Tensor& space,float tanfovx,float tanfovy,
torch::Tensor& rgbs){
    //torch::Device device(torch::kCUDA);
    int res=1024;
    auto float_opts = position.options().dtype(torch::kFloat32);
    torch::Device device(torch::kCUDA);
    //torch::Tensor rgbs = torch::full({res, res, 3}, 0.0, torch::dtype(torch::kFloat32).device(device));
    //CreateScene(d_positions, d_orientations,  d_scales,d_covs,gaussians.size(),1024,d_env_rgbs);
    
    CreateScene(
        position.contiguous().data_ptr<float>(),
        orientation.contiguous().data_ptr<float>(),
        scale.contiguous().data_ptr<float>(),
        cov.contiguous().data_ptr<float>(),
        magnitude.contiguous().data_ptr<float>(),
        albedo.contiguous().data_ptr<float>(),
        position.size(0),
        rgbs.contiguous().data_ptr<float>(),
        1024,
        env_rgbs.contiguous().data_ptr<float>(),
        view_mat.contiguous().data_ptr<float>(),
        space.contiguous().data_ptr<float>(),
        tanfovx,
        tanfovy
    );
    
    
    //return rgbs;
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