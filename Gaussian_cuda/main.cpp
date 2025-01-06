#include<vector>
#include<string>
#include"gaussian.h"
#include"utils.h"
#include"envmap.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"/home/qinhaoran/preprocess/stb/stb_image_write.h"
#include"/home/qinhaoran/preprocess/stb/stb_image.h"
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

int cnt_gs=0,pix_f=0,cnt_val=0,kdgs=0;
void convertAndSaveImage(const float* rgbs, int width, int height) {
    // 创建一个保存图像数据的数组
    unsigned char* image_data = new unsigned char[width * height * 3];  // 3 是因为 RGB 有 3 个通道

    // 将 Selas::float3 转换为 unsigned char
    for (int i = 0; i < width * height; ++i) {
        image_data[i * 3] = static_cast<unsigned char>(rgbs[3*i] * 255.0f);     // Red
        image_data[i * 3 + 1] = static_cast<unsigned char>(rgbs[3*i+1] * 255.0f); // Green
        image_data[i * 3 + 2] = static_cast<unsigned char>(rgbs[3*i+2] * 255.0f); // Blue
    }

    // 保存图像为 PNG 文件
    stbi_write_png("/home/qinhaoran/cuda_output/output_image.png", width, height, 3, image_data, width * 3);

    // 释放内存
    delete[] image_data;
}

float* loadImageAndConvertToFloat3(const std::string& file_path, int width, int height) {
    // 读取图片数据，强制为RGB三通道
    int channels;
    unsigned char* image_data = stbi_load(file_path.c_str(), &width, &height, &channels, 3);  // 3表示RGB

    if (!image_data) {
        std::cerr << "Failed to load image: " << file_path << std::endl;
        return {};
    }

    // 创建保存 Selas::float3 的 vector
    float* rgbs=(float*)malloc(width * height*3*sizeof(float));

    // 将 unsigned char 转换为 Selas::float3，范围 [0, 255] -> [0.0f, 1.0f]
    for (int i = 0; i < width * height; ++i) {
        rgbs[i*3] = image_data[i * 3] / 255.0f;
        rgbs[i*3+1] = image_data[i * 3 + 1] / 255.0f;
        rgbs[i*3+2] = image_data[i * 3 + 2] / 255.0f;
        //rgbs[i*3] = Selas::float3(r, g, b);
        //printf("r%f\n",r);
    }

    // 释放图片数据
    stbi_image_free(image_data);

    return rgbs;
}




int main(){
    /*
    std::string pik1="/home/qinhaoran/preprocess/output/sparse5.png";
    std::string pik2="/home/qinhaoran/preprocess/output/sparse6.png";
    std::vector<Selas::float3> pic1=loadImageAndConvertToFloat3(pik1,1000,1000);
    std::vector<Selas::float3> pic2=loadImageAndConvertToFloat3(pik2,1000,1000);
    for(int i=0;i<1000;i++)
    for(int j=0;j<1000;j++){
    if(std::abs(pic1[i*1000+j].x-pic2[i*1000+j].x)>0.001){
            printf("%f %f %f\n",pic1[i*1000+j].x,pic1[i*1000+j].y,pic1[i*1000+j].z);
            printf("%f %f %f\n",pic2[i*1000+j].x,pic2[i*1000+j].y,pic2[i*1000+j].z);

            printf("%d %d\n",i,j);
            return 0;
        }
    }
    */
    
    std::string env_path="/home/qinhaoran/upload/IndoorEnvironmentHDRI001_8K-TONEMAPPED.jpg";
    float* env_rgbs=loadImageAndConvertToFloat3(env_path,8192,4096);
    //Env_map env_map(env_rgbs,8192,4096);

    std::string input_path="/home/qinhaoran/preprocess/processed/gs_golf.npz";
    std::string input_path1="/data/processed/gs_bunny_new.npz";
    std::string input_path2="/data/qinhaoran/processed/gs_sphere.npz";
    std::string input_path3="/data/qinhaoran/processed/gs_new_bunny.npz";
    
    std::vector<Gaussian> gaussians=loadGaussiansFromNPZ(input_path3);
    printf("111\n");
    float* d_positions, * d_orientations,  * d_scales,  * d_covs;
    copyGaussiansToGPU(gaussians,d_positions, d_orientations,  d_scales,  d_covs);
    printf("finishedmoving\n");
    int res=1024;
    float* rgbs,*h_rgbs;
    h_rgbs=(float*)malloc(res*res*3*sizeof(float));

    float* d_env_rgbs;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_env_rgbs, 8192*4096 *3* sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_env_rgbs, env_rgbs, 8192*4096 *3* sizeof(float),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&rgbs, res*res *3* sizeof(float)));
    CreateScene(d_positions, d_orientations,  d_scales,d_covs,gaussians.size(),rgbs,res,d_env_rgbs);
    //return 0;
    CHECK_CUDA_ERROR(cudaMemcpy(h_rgbs, rgbs, res*res*3*sizeof(float),cudaMemcpyDeviceToHost));

    convertAndSaveImage(h_rgbs,res,res);
    
    return 0;
    
}