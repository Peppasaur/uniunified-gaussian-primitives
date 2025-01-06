#include <cmath>
#include <functional>
#include"utils.h"
#include"gaussian.h"
#include"ray.h"
#include"collide.h"

bool intersects(const Bounding_box& a,const Bounding_box& b) {
        return (a.min_x <= b.max_x && a.max_x >= b.min_x) &&
               (a.min_y <= b.max_y && a.max_y >= b.min_y) &&
               (a.min_z <= b.max_z && a.max_z >= b.min_z);
    }



const float EPSILON = 1e-6;
const int MAX_ITERATIONS = 100;

