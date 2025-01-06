#include"scene.h"
#include<iostream>
#include<ctime>
#include<random>
#include <cstdlib> // 包含 exit 函数的头文件
#include <thread>
#include<future>
#include<pthread.h>
#include<stack>
#include <chrono>
float generateRandomNumber(float x,float y) {
    // 创建随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());

    // 创建 [0, 1] 范围的均匀分布
    std::uniform_real_distribution<> dis(x, y);

    // 返回生成的随机数
    return dis(gen);
}
float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float solve(std::vector<Gaussian*>& gaussians, float t_start, float target_cdf,Ray ray,int iter) {
    // 初始化：可以选择区间中点作为初始值
    float t = t_start;

    // 定义目标函数 f(t) 和 f'(t)
    auto f = [&gaussians, target_cdf,t_start,&ray](float t) -> float {
        float cdf = 0.0f;
        for (auto& gaussian : gaussians) {
            // 计算光线穿过每个高斯的累积分布函数
            cdf += gaussian->gaussianRayIntegral(t_start,t,ray); // 假设这里有高斯的 CDF 函数
        }
        return cdf - target_cdf;
    };

    auto f_prime = [&gaussians,&ray](float t) -> float {
        float pdf = 0.0f;
        for (auto& gaussian : gaussians) {
            // 计算每个高斯的概率密度函数 (PDF)
            pdf += gaussian->getGaussian(t,ray); // 假设这里有高斯的 PDF 函数
        }
        return pdf;
    };

    // 牛顿迭代法
    for (int i = 0; i < iter; ++i) {
        float ft = f(t);       // 当前的函数值
        float ft_prime = f_prime(t); // 当前的导数

        if (std::fabs(ft) < 0.001) {
            // 如果函数值已经足够接近零，停止迭代
            break;
        }

        if (ft_prime == 0.0f) {
            // 避免除以零的情况，如果导数为零，则退出
            break;
        }

        // 牛顿-拉夫森迭代公式
        t = t - ft / ft_prime;
    }

    return t;
}

Eigen::Matrix3f createOrthogonalMatrix(const Eigen::Vector3f& input_vector) {
    // 确保输入向量是单位向量
    Eigen::Vector3f normalized_vector = input_vector.normalized();

    // 创建一个矩阵，第三列为输入向量
    Eigen::Matrix3f orthogonal_matrix;
    orthogonal_matrix.col(2) = normalized_vector;

    // 创建一个正交的第一个向量，任意选择一个不平行的向量
    Eigen::Vector3f arbitrary_vector(1, 0, 0); // 选择x轴方向作为初始向量
    if (normalized_vector.isApprox(arbitrary_vector)) {
        arbitrary_vector = Eigen::Vector3f(0, 1, 0); // 如果与x轴平行，则选择y轴
    }

    // 计算第二个向量
    Eigen::Vector3f first_vector = normalized_vector.cross(arbitrary_vector).normalized();
    // 计算第一个向量的正交向量
    Eigen::Vector3f second_vector = normalized_vector.cross(first_vector).normalized();

    // 将正交向量放入矩阵
    orthogonal_matrix.col(0) = first_vector;
    orthogonal_matrix.col(1) = second_vector;

    return orthogonal_matrix;
}

Eigen::Vector3f Gaussian::transdir(Eigen::Vector3f dir,Eigen::Matrix3f orien){
    Eigen::Vector3f dir_tan=orien.transpose()*dir;
    std::swap(dir_tan(2),dir_tan(1));
    return dir_tan;
}

Eigen::Vector3f Gaussian::invtransdir(Eigen::Vector3f dir_tan,Eigen::Matrix3f orien){
    std::swap(dir_tan(2),dir_tan(1));
    Eigen::Vector3f dir=orien*dir_tan;
    return dir;
}

Eigen::Vector3f intersectRayWithPlane(const Eigen::Vector3f& rayOrigin, 
                                              const Eigen::Vector3f& rayDir, 
                                              const Eigen::Vector3f& planeNormal, 
                                              const Eigen::Vector3f& pointOnPlane) {
    // 计算射线方向与平面法向量的点积
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

void directionToSpherical(Eigen::Vector3f direction, float& theta, float& phi) {
    // 计算 r 的值，r 是方向向量的长度

    // 计算 theta（倾角），与 z 轴的夹角
    theta = std::asin(direction(2) );

    // 计算 phi（方位角），在 xy 平面上的角度
    phi = std::atan2(direction(1), direction(0));
}

Eigen::Vector3f get_upper_hemisphere(float x, float y, float radius = 0.5) {
    Eigen::Vector3f result;

    // 如果 x^2 + y^2 超过了半径平方，说明点不在球体范围内
    if (x * x + y * y > radius * radius) {
        std::cerr << "Point is outside of the sphere projection." << std::endl;
        return {0, 0, 0}; // 返回一个空的向量
    }

    // 计算 z 坐标
    float z = std::sqrt(radius * radius - x * x - y * y);

    result(0) = x;
    result(1) = y;
    result(2) = z;

    return result*2;
}

Eigen::Vector3f Gaussian::tan_to_world(Eigen::Vector3f src){
    Eigen::Vector3f res;
    res=src-(this->position);
    res=(this->orientation).transpose()*res;
    return res;
}

Selas::float3 Scene::shading(Ray& ray,int depth,int max_depth,int start,float sca,Gaussian* start_gs,int start_row){
    auto start_time = std::chrono::high_resolution_clock::now();
    Selas::float3 dark(0,0,0);
    Selas::float3 ambient((float)100/255,(float)100/255,(float)100/255);
    
    if(depth>=max_depth){
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        if(start_row == 500)printf("shadetime1 %lf\n",elapsed.count());
        return ambient;
    }

    
    float t0=sca*5,t1=100,cdf=0;
    kdtree.root->intersectBound(ray, t0, t1);
    float u=generateRandomNumber(0,1);

    auto start_time2 = std::chrono::high_resolution_clock::now();
    Eigen::Vector3f accum_norm(0,0,0);
    Gaussian* res=sample(kdtree.root,ray,t0,t1,u,cdf,accum_norm,start,start_row);
    accum_norm=accum_norm.normalized();
    Eigen::Matrix3f accum_orien=createOrthogonalMatrix(accum_norm);

    auto end_time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end_time2 - start_time2;
    if(start_row == 500)printf("sampletime %lf\n",elapsed2.count());
    
    if(res!=nullptr){
        cnt_val++;
        //printf("depth%d res->id%d\n",depth,res->id);
        Selas::BsdfSample sample;
        
        Eigen::Vector3f dot=(res->orientation).transpose()*(-ray.direction);

        if(dot(2)<0){
            res->orientation(0,2)=- res->orientation(0,2);
            res->orientation(1,2)=- res->orientation(1,2);
            res->orientation(2,2)=- res->orientation(2,2);
        }
        
        //if(dot(2)<0)return ambient;
        Eigen::Vector3f dir_tan=res->transdir(-ray.direction,accum_orien);
        Selas::float3 v(dir_tan(0),dir_tan(1),dir_tan(2));
        
        
        bool suc=Selas::SampleDisney(&(this->sampler), (this->surfaces)[res->id], v, 0, sample);
        
        sample.wi.x=-v.x;
        sample.wi.y=v.y;
        sample.wi.z=-v.z;
        
        Eigen::Vector3f dir(sample.wi.x,sample.wi.y,sample.wi.z);
        dir=res->invtransdir(dir,accum_orien);

        //Eigen::Vector3f mid=((-ray.direction)+dir).normalized();
        //std::cout<<(res->orientation.transpose()*mid)<<std::endl;
        Eigen::Vector3f norm(res->orientation(0,2),res->orientation(1,2),res->orientation(2,2));
        Eigen::Vector3f src=intersectRayWithPlane(ray.source,ray.direction,norm,res->position);
        Eigen::Vector3f src_tan= res->tan_to_world(src);
        //printf("src_tan%f\n",src_tan(2));
        //Eigen::Vector3f curv_norm=get_upper_hemisphere(src_tan(0),src_tan(1));
        //norm=(res->orientation)*curv_norm;

        Ray newray(src,dir);
        Selas::float3 rgb(0,0,0);
        rgb=shading(newray,depth+1,max_depth,res->id,res->scale(0),res,start_row);
        //printf("refx%f refy%f refz%f\n",sample.reflectance.x,sample.reflectance.y,sample.reflectance.z);
        //printf("rever%f\n",sample.reversePdfW);
        Selas::float3 ref(0.8,0.8,0.8);
        rgb=rgb;
        //rgb=rgb*sample.reflectance;
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        if(start_row == 500)printf("shadetime2 %lf\n",elapsed.count());

        return rgb;
        
    }
    

    Eigen::Vector3f dest=ray.source+ray.direction*t1;
    //if(dest(0)>0.95||dest(0)<0.05||dest(1)>0.95||dest(1)<0.05||dest(2)>0.95)return back_color;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    if(start_row == 500)printf("shadetime3 %lf\n",elapsed.count());
    //if(dest(0)>0.95)return back_color;
    //return ambient;
    float theta=0,phi=0;
    int x=0,y=0;
    directionToSpherical(ray.direction, theta, phi);
    this->env_map.sphericalToImageCoords(theta, phi,x,y);
    //std::cout<<ray.direction<<std::endl;
   // printf("theta%f phi%f\n",theta,phi);
    //printf("x%d y%d\n",x,y);
    //if((x+y)%3==0)return Selas::float3(1,0,0);
    //if((x+y)%3==1)return Selas::float3(0,1,0);
    //if((x+y)%3==2)return Selas::float3(0,0,1);
    return this->env_map.rgbs[x*(this->env_map.width)+y];

}
/*
std::vector<Selas::float3> Scene::render_scene(int res,float focal,int sample_per_pix){
    std::vector<Selas::float3>rgbs(res*res);
    for(int i=0;i<res;i++){
        for(int j=0;j<res;j++){
            float offset=(float)1/(2*res);
            //Eigen::Vector3f pix_pos((float)i/res+offset,(float)j/res+offset,0);
            //Eigen::Vector3f source(0.5,0.5,-focal);
            Eigen::Vector3f pix_pos(1,(float)i / res + offset , (float)j / res + offset );
            Eigen::Vector3f source(1+focal, 0.5,0.5);

            Eigen::Vector3f direction=(pix_pos-source).normalized();
            //printf("dir%f %f %f\n",pix_pos(0),pix_pos(1),pix_pos(2));
            Ray ray(source,direction);
            auto start_time = std::chrono::high_resolution_clock::now();
            rgbs[i*res+j]=shading(ray,0,1,-1,0,nullptr,500);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            printf("shadetime %lf\n",elapsed.count());
            if(rgbs[i*res+j].x!=1)printf("%d %d %f %f %f\n",i,j,rgbs[i*res+j].x,rgbs[i*res+j].y,rgbs[i*res+j].z);
        }
    }
    return rgbs;
}
*/

std::vector<Selas::float3> Scene::render_scene(int res, float focal, int sample_per_pix ) {
    std::vector<Selas::float3> rgbs(res * res);
    std::vector<std::future<void>> futures; // 存储每个线程的future对象
    //int num_threads = std::thread::hardware_concurrency();
    int num_threads =20;
    printf("num_threads%d\n",num_threads);

    double avgpix=0,avgshade=0;
    // 每个线程要处理的行数
    int part_size = res / num_threads;
    
    // 渲染指定范围的像素
    auto render_pixel_range = [this, &rgbs, res, focal, sample_per_pix,num_threads,&avgpix,&avgshade](int start_row ) {
        printf("%d\n",pthread_self());
        for (int i = start_row; i < res; i+=num_threads) {
            auto start_time1 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < res; ++j) {
                auto start_time2 = std::chrono::high_resolution_clock::now();
                float offset = (float)1 / (2 * res);
                float biasx = generateRandomNumber(0, offset) - (offset / 2);
                float biasy = generateRandomNumber(0, offset) - (offset / 2);
                //Eigen::Vector3f pix_pos((float)i / res + offset + biasx, (float)j / res + offset + biasy, 0);
                //Eigen::Vector3f source(0.5, 0.5, -focal);
                //Eigen::Vector3f pix_pos((float)i / res + offset + biasx, (float)j / res + offset + biasy, 0);
                //Eigen::Vector3f source(0.5, 0.5, -focal);
                Eigen::Vector3f pix_pos(0,(float)i / res + offset+biasx , (float)j / res + offset+biasy );
                Eigen::Vector3f source(-focal, 0.5,0.5);

                Eigen::Vector3f direction = (pix_pos - source).normalized();
                Ray ray(source, direction);
                rgbs[i * res + j] = Selas::float3(0, 0, 0);
                auto start_time = std::chrono::high_resolution_clock::now();
                //printf("sample_per_pix%d\n",sample_per_pix);
                for (int k = 0; k < sample_per_pix; ++k) {
                    rgbs[i * res + j] = rgbs[i * res + j] + shading(ray, 0, 2, -1, 0, nullptr,start_row);
                }
                if(i==300&&j==550){
                    printf("rgbs%f %f %f\n",rgbs[i * res + j].x,rgbs[i * res + j].y,rgbs[i * res + j].z);
                    //exit(0);
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_time - start_time;
                if(start_row == 500)printf("shadetime %lf\n",elapsed.count());
                avgshade+=elapsed.count();

                rgbs[i * res + j] = rgbs[i * res + j] / (float)sample_per_pix;

                if (rgbs[i * res + j].x != 1) {
                    if(start_row == 0)printf("%d %d %f %f %f\n", i, j, rgbs[i * res + j].x, rgbs[i * res + j].y, rgbs[i * res + j].z);
                }
                auto end_time2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed2 = end_time2 - start_time2;
                //printf("pixtime %lf\n",elapsed2.count());
                avgpix+=elapsed2.count();
            }
            auto end_time1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed1 = end_time1 - start_time1;
            //printf("rowtime %lf\n",elapsed1.count());
        }
    };

    // 根据线程数将任务分配给每个线程
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * part_size;
        int end_row = (i == num_threads - 1) ? res : start_row + part_size;
        futures.push_back(std::async(std::launch::async, render_pixel_range, i));
    }

    // 等待所有线程完成
    for (auto& future : futures) {
        future.get();
    }
    avgpix/=1000000;
    avgshade/=1000000;
    printf("avgpix%lf\n",avgpix);
    printf("avgshade%lf\n",avgshade);
    return rgbs;
}

Gaussian* Scene::sample(KDNode* node, Ray& ray, float t0, float t1, float u, float &cdf,Eigen::Vector3f& accum_norm,int startid,int start_row) {
    //printf("333\n");
    if (!node->is_leaf) {
        // 前向到后向遍历
        
        auto start_time = std::chrono::high_resolution_clock::now();
        KDNode* frontNode = node->front(ray);
        KDNode* backNode = node->back(ray);
        float t0Front = t0, t1Front = t1, t0Back = t0, t1Back = t1;

        // 检查与前子节点的交集
        
        frontNode->intersectBound(ray, t0Front, t1Front);
        //printf("t0f%f t1f%f\n",t0Front,t1Front);
        
        //printf("t0Front%f t1Front%f\n",t0Front,t1Front);
        Gaussian* result=0;
        if(t0Front<t1Front)result = sample(frontNode, ray, t0Front, t1Front, u, cdf,accum_norm,startid,start_row);
        if (result!=0) {
            //printf("result%d\n",result);
            return result;
        }

        // 检查与后子节点的交集
        backNode->intersectBound(ray, t0Back, t1Back);
        //printf("t0Back%f t1Back%f\n",t0Back,t1Back);
        if(t0Back<t1Back)result = sample(backNode, ray, t0Back, t1Back, u, cdf,accum_norm,startid,start_row);
        if (result!=0) {
            //printf("result%d\n",result);
            return result;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        //if(start_row == 500)printf("recurtime %lf\n",elapsed.count());
    } else {
        //printf("t0%f t1%f\n",t0,t1);
        //printf("leaf\n");
        // 处理叶子节点
        float integer_sum=0;
        //for (auto& p : node->gaussians) integer_sum+=p->gaussianRayIntegral(t0,t1,ray);
        if(node->gaussians.size()>0){
            //printf("gs%d\n",node->gaussians.size());
            //if(integer_sum>0)printf("integer_sum%f\n",integer_sum);
        }
        //if (node->gaussians.size() == 1 || cdf + 0.5 * integer_sum < 1) {
        if (1) {
            // 案例（1）和（2）
            int cnt=0;
            for (auto& p : node->gaussians) {
                cnt++;
                if(startid!=p->id){
                    
                    auto start_time = std::chrono::high_resolution_clock::now();
                    float integer=p->gaussianRayIntegral(t0, t1,ray);

                    Eigen::Vector3f norm(p->orientation(0,2),p->orientation(1,2),p->orientation(2,2));

                    Eigen::Vector3f surpoint=intersectRayWithPlane(ray.source,ray.direction,norm,p->position);
                    float dist=(surpoint-p->position).norm();
                    accum_norm+=norm*(1/dist/dist);
                    

                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_time - start_time;
                    //if(start_row == 500)printf("intetime %lf\n",elapsed.count());
                    //std::cout << " execution time: " << elapsed.count() << " seconds" << std::endl;
                    //std::cout << "耗时: " << elapsed << " 秒" << std::endl;
                    cdf += 0.5 * integer;
                    cnt_gs++;
                    //printf("gaussian%d ",p->id);

                    //if(integer>0.01)printf("inte %f\n",integer);
                    if (u < cdf) {
                        //printf("cnt %d\n",cnt);
                        //printf("id%d\n",p.id);
                        return p; // 返回采样的高斯对象和CDF
                    }
                }
            }
            //printf("\ncnt %d cdf %f\n",cnt,cdf);
        } else {
            // 案例（3），需要处理模糊节点
            auto segments = node->partition(ray, t0, t1);
            
            
            for (auto& seg : segments) {
                float segment_sum=0;
                for(auto& p:seg.gaussians)segment_sum+=p->gaussianRayIntegral(t0,t1,ray);
                if (seg.gaussians.size() == 1 || cdf + 0.5 * segment_sum < 1) {
                    for (auto& p : seg.gaussians) {
                        return p;
                        cdf += 0.5 * 10;
                        //p->gaussianRayIntegral(seg.t_start, seg.t_end,ray);
                        if (u < cdf) {
                            //printf("man\n");
                            //printf("id%d\n",p.id);
                            return p; // 返回采样的高斯对象和CDF
                        }
                    }
                } else {
                    // 处理模糊节点的消歧义
                    float t_u = solve(seg.gaussians, seg.t_start, u - cdf,ray,500);
                    float u_seg = lerp(cdf, u, generateRandomNumber(0,1)); // 生成一个新的随机数
                    for (auto& p : seg.gaussians) {
                        cdf += 0.5 * p->gaussianRayIntegral(seg.t_start, t_u,ray);
                        if (u_seg < cdf) {
                            //printf("man\n");
                            //printf( "id%d\n",p.id);
                            return p; // 返回采样的高斯对象和CDF
                        }
                    }
                }
            }
        }
    }
    //printf("man\n");
    return 0; // 在没有节点的情况下返回空结果
}