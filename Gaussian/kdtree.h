#include<vector>
#include"ray.h"
#include"utils.h"
#include"gaussian.h"// kd-tree 节点类，包含空间分割和包围盒
#include"collide.h"

extern int kdgs;
struct Event {
        float t;
        Gaussian* primitive; // 与哪个原语相关
        bool isEntry;  // true 表示进入，false 表示离开
    };
struct Segment {
    float t_start, t_end;
    std::vector<Gaussian*> gaussians; // 保存与该区间相交的原语索引
};


struct KDNode {
    Bounding_box bounding_box;         // 当前节点的包围盒
    std::vector<Gaussian*> gaussians;   // 当前节点的高斯对象（仅限叶子节点）
    KDNode* left = nullptr;            // 左子节点
    KDNode* right = nullptr;           // 右子节点
    bool is_leaf=0;
    int split;

    KDNode(const Bounding_box& box,std::vector<Gaussian*>gs) : bounding_box(box),gaussians(gs) {}

    // 释放节点内存
    ~KDNode() {
        delete left;
        delete right;
    }

    KDNode* front(Ray ray);
    KDNode* back(Ray ray);
    void intersectBound(Ray ray,float &t0,float &t1);
    std::vector<Segment> partition(const Ray& ray, float t0, float t1);
};

// KDTree类，用于管理kd-tree的构建和操作
class KDTree {
public:
    KDNode* root = nullptr;  // kd-tree的根节点

    // 构造函数，初始化根节点为空
    KDTree() = default;

    // 析构函数，释放根节点的内存
    ~KDTree() {
        delete root;
    }

    // 构建kd-tree的成员函数
    void build(std::vector<Gaussian>& gaussians, const Bounding_box& bounds, int maxDepth = 15) {
        std::vector<Gaussian*> gss;
        for(auto& gs:gaussians)gss.push_back(&gs);
        root = buildKDTree(gss, bounds, 0, maxDepth);
    }

private:
    // 递归构建kd-tree的私有成员函数
    KDNode* buildKDTree(std::vector<Gaussian*>& gaussians, const Bounding_box& bounds, int depth, int maxDepth);
};
