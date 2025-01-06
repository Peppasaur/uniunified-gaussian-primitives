#include<vector>
#include<string>
#include <chrono>
#include"gaussian.h"
#include"scene.h"
#include"utils.h"
#include"envmap.h"
#include"../MathLib/FloatStructs.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"/home/qinhaoran/preprocess/stb/stb_image_write.h"
#include"/home/qinhaoran/preprocess/stb/stb_image.h"
int cnt_gs=0,pix_f=0,cnt_val=0,kdgs=0;
void convertAndSaveImage(const std::vector<Selas::float3>& rgbs, int width, int height) {
    // 创建一个保存图像数据的数组
    unsigned char* image_data = new unsigned char[width * height * 3];  // 3 是因为 RGB 有 3 个通道

    // 将 Selas::float3 转换为 unsigned char
    for (int i = 0; i < width * height; ++i) {
        image_data[i * 3] = static_cast<unsigned char>(rgbs[i].x * 255.0f);     // Red
        image_data[i * 3 + 1] = static_cast<unsigned char>(rgbs[i].y * 255.0f); // Green
        image_data[i * 3 + 2] = static_cast<unsigned char>(rgbs[i].z * 255.0f); // Blue
    }

    // 保存图像为 PNG 文件
    stbi_write_png("/home/qinhaoran/preprocess/output/output_image.png", width, height, 3, image_data, width * 3);

    // 释放内存
    delete[] image_data;
}

std::vector<Selas::float3> loadImageAndConvertToFloat3(const std::string& file_path, int width, int height) {
    // 读取图片数据，强制为RGB三通道
    int channels;
    unsigned char* image_data = stbi_load(file_path.c_str(), &width, &height, &channels, 3);  // 3表示RGB

    if (!image_data) {
        std::cerr << "Failed to load image: " << file_path << std::endl;
        return {};
    }

    // 创建保存 Selas::float3 的 vector
    std::vector<Selas::float3> rgbs(width * height);

    // 将 unsigned char 转换为 Selas::float3，范围 [0, 255] -> [0.0f, 1.0f]
    for (int i = 0; i < width * height; ++i) {
        float r = image_data[i * 3] / 255.0f;
        float g = image_data[i * 3 + 1] / 255.0f;
        float b = image_data[i * 3 + 2] / 255.0f;
        rgbs[i] = Selas::float3(r, g, b);
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
    std::vector<Selas::float3> env_rgbs=loadImageAndConvertToFloat3(env_path,8192,4096);
    Env_map env_map(env_rgbs,8192,4096);

    std::string input_path="/home/qinhaoran/preprocess/processed/gs_golf.npz";
    std::string input_path1="/data/processed/gs_sphere.npz";
    std::string input_path2="/data/qinhaoran/processed/gs_sphere.npz";
    std::string input_path3="/data/qinhaoran/processed/gs_new_bunny.npz";
    std::vector<Gaussian> gaussians=loadGaussiansFromNPZ(input_path3);
    printf("111\n");
    //return 0;
    Bounding_box bound={0,0,0,1,1,1};
    Selas::float3 back_color(1,1,1);
    
    std::vector<Selas::float3>rgbs;
    int res=1000;
    //printf("222\n");
    //return 0;
    std::string mat_path="/home/qinhaoran/upload/BSDF1.json";
    Selas::Material material;
    material.LoadFromJson(mat_path);

    printf("ready\n");
    std::vector<Selas::SurfaceParameters> surfaces(gaussians.size());
    for(int i=0;i<gaussians.size();i++){
        surfaces[i].createParam(material,gaussians[i]);
    }
    Selas::CSampler sampler;
    sampler.Initialize(0);

    

    Scene scene(gaussians,bound,back_color,material,surfaces,sampler,env_map);
    auto start_time = std::chrono::high_resolution_clock::now();
    rgbs=scene.render_scene(res,1,1);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    printf("rendertime %lf\n",elapsed.count());
    convertAndSaveImage(rgbs,res,res);
    printf("cnt_gs%d cnt_val%d kdgs%d\n",cnt_gs,cnt_val,kdgs);
    
}