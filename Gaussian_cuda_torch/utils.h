#ifndef UTILS_H
#define UTILS_H

#include<vector>

struct Bounding_box{
    float min_x,min_y,min_z,max_x,max_y,max_z;
};



bool intersects(const Bounding_box& a,const Bounding_box& b);
#endif 