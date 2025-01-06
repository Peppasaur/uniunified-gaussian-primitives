#include"kdtree.h"
#include"utils.h"
#include"gaussian.h"
#include <future>
#include <cstdlib> // 包含 exit 函数的头文件

/*
bool ellipsoid_collision(Gaussian* gs,const Bounding_box& box){
    fcl::Vector3f escale;
    
    for(int i=0;i<3;i++){
        float sca=gs->scale(i);
        float exp_bound=gs->magnitude*0.1*pow(sca,3)*pow(2*M_PI,(float)3/2);
        float bound_2=-2*log(exp_bound)*pow(sca,2);
        float bound=sqrt(bound_2);
        escale(i)=bound;
    }
    
    fcl::Vector3f bcenter((box.min_x+box.max_x)/2,(box.min_y+box.max_y)/2,(box.min_z+box.max_z)/2);
    fcl::Vector3f bsize(box.max_x-box.min_x,box.max_y-box.min_y,box.max_z-box.min_z);
    fcl::Matrix3f rotation;  
    rotation << gs->orientation(0,0), gs->orientation(0,1), gs->orientation(0,2),
                    gs->orientation(1,0), gs->orientation(1,1), gs->orientation(1,2),
                    gs->orientation(2,0), gs->orientation(2,1), gs->orientation(2,2);  
    fcl::Vector3f pos(gs->position(0),gs->position(1),gs->position(2));
    bool res=collision(pos, escale,rotation,bcenter, bsize);
    return res;
}
*/

std::pair<std::vector<Gaussian*>, std::vector<Gaussian*>> processGaussiansPart(
    const std::vector<Gaussian*>& gaussians_part,
    const Bounding_box& leftBounds,
    const Bounding_box& rightBounds) 
{
    std::vector<Gaussian*> leftGaussians;
    std::vector<Gaussian*> rightGaussians;

    for (const auto& gaussian : gaussians_part) {
        if (intersects(gaussian->bbox, leftBounds)) {
            leftGaussians.push_back(gaussian);
        }
        if (intersects(gaussian->bbox, rightBounds)) {
            rightGaussians.push_back(gaussian);
        }
    }

    // 返回左高斯体和右高斯体
    return {leftGaussians, rightGaussians};
}

void processGaussiansParallel(
    const std::vector<Gaussian*>& gaussians,
    const Bounding_box& leftBounds,
    const Bounding_box& rightBounds,
    std::vector<Gaussian*>& leftGaussians,
    std::vector<Gaussian*>& rightGaussians,
    int num_threads)
{
    int total_size = gaussians.size();
    int part_size = total_size / num_threads;
    std::vector<std::future<std::pair<std::vector<Gaussian*>, std::vector<Gaussian*>>> > futures;

    // 为每个线程创建一个任务
    for (int i = 0; i < num_threads; ++i) {
        auto start_it = gaussians.begin() + i * part_size;
        auto end_it = (i == num_threads - 1) ? gaussians.end() : start_it + part_size;

        // 创建一个线程并异步执行
        futures.push_back(std::async(std::launch::async, processGaussiansPart,
                                     std::vector<Gaussian*>(start_it, end_it),
                                     std::ref(leftBounds), std::ref(rightBounds)));
    }

    // 收集每个线程的结果
    for (auto& future : futures) {
        auto part_results = future.get();
        // 合并每个线程的结果到主线程中的结果向量
        leftGaussians.insert(leftGaussians.end(), part_results.first.begin(), part_results.first.end());
        rightGaussians.insert(rightGaussians.end(), part_results.second.begin(), part_results.second.end());
    }
}

KDNode* KDNode::front(Ray ray){
    if(ray.direction(this->split)>0)return this->left;
    return this->right;
}

KDNode* KDNode::back(Ray ray){
    if(ray.direction(this->split)<0)return this->left;
    return this->right;
}

void KDNode::intersectBound(Ray ray,float &t0,float &t1){
            // x方向
    float invDirX = 1.0f / ray.direction.x(); // 方向倒数，避免除零
    float tminX = (bounding_box.min_x - ray.source.x()) * invDirX;
    float tmaxX = (bounding_box.max_x - ray.source.x()) * invDirX;
    if (invDirX < 0.0f) std::swap(tminX, tmaxX); // 确保tminX < tmaxX
    //printf("tminX%f tmaxX%f\n",tminX,tmaxX);
    t0 = std::max(t0, tminX);
    t1 = std::min(t1, tmaxX);
    
    // y方向
    float invDirY = 1.0f / ray.direction.y();
    float tminY = (bounding_box.min_y - ray.source.y()) * invDirY;
    float tmaxY = (bounding_box.max_y - ray.source.y()) * invDirY;
    if (invDirY < 0.0f) std::swap(tminY, tmaxY);
    //printf("tminY%f tmaxY%f\n",tminY,tmaxY);
    t0 = std::max(t0, tminY);
    t1 = std::min(t1, tmaxY);
    
    // z方向
    float invDirZ = 1.0f / ray.direction.z();
    float tminZ = (bounding_box.min_z - ray.source.z()) * invDirZ;
    float tmaxZ = (bounding_box.max_z - ray.source.z()) * invDirZ;
    if (invDirZ < 0.0f) std::swap(tminZ, tmaxZ);
    //printf("tminZ%f tmaxZ%f\n",tminZ,tmaxZ);
    t0 = std::max(t0, tminZ);
    t1 = std::min(t1, tmaxZ);

}
int cnt=0;
KDNode* KDTree::buildKDTree(std::vector<Gaussian*>& gaussians, const Bounding_box& bounds, int depth, int maxDepth) {
    //printf("kdstart\n");
    // 递归结束条件：如果达到最大深度或没有高斯对象
    if (depth >= maxDepth) {
        KDNode* node= new KDNode(bounds,gaussians);
        node->is_leaf=1;
        kdgs+=gaussians.size();
        return node;
    }

    // 选择分割轴，按深度轮换分割轴（0: x, 1: y, 2: z）
    int axis = depth % 3; // 0 for x, 1 for y, 2 for z
    float splitValue;

    if (axis == 0) { // x 轴
        splitValue = 0.5f * (bounds.min_x + bounds.max_x);
    } else if (axis == 1) { // y 轴
        splitValue = 0.5f * (bounds.min_y + bounds.max_y);
    } else { // z 轴
        splitValue = 0.5f * (bounds.min_z + bounds.max_z);
    }

    // 定义子节点的包围盒
    Bounding_box leftBounds = bounds;
    Bounding_box rightBounds = bounds;

    if (axis == 0) {
        leftBounds.max_x = splitValue;
        rightBounds.min_x = splitValue;
    } else if (axis == 1) {
        leftBounds.max_y = splitValue;
        rightBounds.min_y = splitValue;
    } else { // axis == 2
        leftBounds.max_z = splitValue;
        rightBounds.min_z = splitValue;
    }

    // 创建当前节点
    KDNode* node = new KDNode(bounds,gaussians);
    printf("cnt%d node%d\n",cnt,gaussians.size());
    cnt++;
    //if(cnt>30)exit(0);
    node->split=axis;

    // 将高斯对象划分到子节点
    std::vector<Gaussian*> leftGaussians, rightGaussians;
    
    if(1){
    int i=0;
    for (const auto& gaussian : gaussians) {
        
        i++;
        /*
        //printf("%d\n",i);
        if (ellipsoid_collision(gaussian,leftBounds)) {
            leftGaussians.push_back(gaussian);
        }
        if (ellipsoid_collision(gaussian,rightBounds)) {
            rightGaussians.push_back(gaussian);
        }
        */
        
        if (intersects(gaussian->bbox,leftBounds)) {
            leftGaussians.push_back(gaussian);
        }
        if (intersects(gaussian->bbox,rightBounds)) {
            rightGaussians.push_back(gaussian);
        }
        
    }
    }
    else{
        int num_threads = std::thread::hardware_concurrency();  // 获取系统线程数
        printf("num_threads%d\n",num_threads);
        processGaussiansParallel(gaussians, leftBounds, rightBounds, leftGaussians, rightGaussians, num_threads);
    }
    
    
    // 递归构建左右子树
    node->left = buildKDTree(leftGaussians, leftBounds, depth + 1, maxDepth);
    node->right = buildKDTree(rightGaussians, rightBounds, depth + 1, maxDepth);

    return node;
}

