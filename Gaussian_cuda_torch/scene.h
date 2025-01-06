#ifndef Scene_H
#define Scene_H
#include<vector>
#include <Eigen/Dense>
#include"../MathLib/FloatStructs.h"
#include"envmap.h"
#include"ray.h"
#include"gaussian.h"
#include"kdtree.h"
#include"../Shading/Disney.h"

extern int cnt_gs,pix_f,cnt_val;

class Scene{
public:
    Selas::float3 back_color;
    Selas::Material material;
    std::vector<Selas::SurfaceParameters> surfaces;
    Selas::CSampler sampler;
    KDTree kdtree;
    std::vector<Selas::float3>render_scene(int res,float focal,int sample_per_pix);
    Gaussian* sample(KDNode* node, Ray& ray, float t0, float t1, float u, float &cdf,Eigen::Vector3f &accum_norm,int startid,int start_row);
    Selas::float3 shading(Ray& ray,int depth,int max_depth,int start,float sca,Gaussian* start_gs,int start_row);
    Env_map env_map;

    Scene(std::vector<Gaussian>& gaussians, const Bounding_box& bounds, Selas::float3 background,Selas::Material material,std::vector<Selas::SurfaceParameters>& surfaces,Selas::CSampler sampler, Env_map env,int maxDepth = 21) 
        : back_color(background),material(material),surfaces(surfaces),sampler(sampler),env_map(env) {  // 使用初始化列表来初始化 back_color
        kdtree.build(gaussians, bounds, maxDepth);  // 调用 kdtree 的 build 函数
    }
};

#endif

