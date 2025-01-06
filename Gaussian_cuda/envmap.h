#ifndef ENV_H
#define ENV_H
#include<vector>
#include"../MathLib/FloatStructs.h"

class Env_map{
public:
    std::vector<Selas::float3>rgbs;
    Env_map(std::vector<Selas::float3>rgbs,int wd,int dt):rgbs(rgbs),width(wd),height(dt){}
    int width;
    int height;
    void sphericalToImageCoords(float theta, float phi,int& x,int& y);
};


#endif