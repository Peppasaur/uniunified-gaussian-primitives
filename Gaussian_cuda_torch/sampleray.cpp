#include"gaussian.h"
#include"kdtree.h"
#include"ray.h"
/*
Gaussian sample(KDNode* node, Ray& ray, float t0, float t1, float u, float &cdf) {
        if (!node->is_leaf()) {
            // 前向到后向遍历
            KDNode* frontNode = node->front(ray);
            KDNode* backNode = node->back(ray);
            float t0Front = t0, t1Front = t1, t0Back = t0, t1Back = t1;

            // 检查与前子节点的交集
            frontNode->intersectBound(ray, t0Front, t1Front);
            if(t0Front<t1Front)Gaussian result = sample(frontNode, ray, t0Front, t1Front, u, cdf);
            if (result) {
                return result;
            }

            // 检查与后子节点的交集
            backNode->intersectBound(ray, t0Back, t1Back);
            if(t0Back<t1Back)result = sample(backNode, ray, t0Back, t1Back, u, cdf);
            if (result) {
                return result;
            }
        } else {
            // 处理叶子节点
            float integer_sum=0;
            for (auto& p : node->gaussians) integer_sum+=p.gaussianRayIntegral(t0,t1,ray);
            if (node->gaussians.size() == 1 || cdf + 0.5 * integer_sum < 1) {
                // 案例（1）和（2）
                for (auto& p : node->gaussians) {
                    cdf += 0.5 * p.gaussianRayIntegral(t0, t1,ray);
                    if (u < cdf) {
                        return p; // 返回采样的高斯对象和CDF
                    }
                }
            } else {
                // 案例（3），需要处理模糊节点
                auto segments = node->partition(ray, t0, t1);
                
                
                for (auto& seg : segments) {
                    float segment_sum=0;
                    for(auto& p:seg.gaussians)segment_sum+=p.gaussianRayIntegral(t0,t1,ray);
                    if (seg.gaussians.size() == 1 || cdf + 0.5 * segment_sum < 1) {
                        for (auto& p : seg.gaussians) {
                            cdf += 0.5 * p.gaussianRayIntegral(seg.t_start, seg.t_end,ray);
                            if (u < cdf) {
                                return p; // 返回采样的高斯对象和CDF
                            }
                        }
                    } else {
                        // 处理模糊节点的消歧义
                        float t_u = solve(seg.gaussians, seg.t_start, u - cdf);
                        float u_seg = lerp(cdf, u, rnd()); // 生成一个新的随机数
                        for (auto& p : seg.gaussians) {
                            cdf += 0.5 * p.gaussianRayIntegral(seg.t_start, t_u);
                            if (u_seg < cdf) {
                                return p; // 返回采样的高斯对象和CDF
                            }
                        }
                    }
                }
            }
        }
    return nullptr // 在没有节点的情况下返回空结果
}
*/