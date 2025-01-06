#include<cmath>
#include"envmap.h"
#include"../MathLib/FloatStructs.h"
using namespace Selas;
void Env_map::sphericalToImageCoords(float theta, float phi,int& x,int& y) {
    // 将经度 theta 从 [-π, π] 映射到 [0, IMAGE_WIDTH-1]
    x = static_cast<int>((theta + (M_PI/2)) / (M_PI) * (this->height - 1));
    
    // 将纬度 phi 从 [-π/2, π/2] 映射到 [0, IMAGE_HEIGHT-1]
    y = static_cast<int>((phi + (M_PI)) / (2*M_PI) * (this->width - 1));
    
    // 保证坐标不越界
    x = std::max(0, std::min(this->height - 1, x));
    y = std::max(0, std::min(this->width - 1, y));

}