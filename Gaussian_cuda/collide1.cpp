
/*
#include"collide.h"
#include "fcl/math/constants.h"
#include "fcl/narrowphase/collision.h"
#include "fcl/narrowphase/collision_object.h"
#include "fcl/narrowphase/distance.h"

bool collision(fcl::Vector3f& ellipsoid_center, fcl::Vector3f& scale,fcl::Matrix3f& rotation,
                                 fcl::Vector3f& box_center, fcl::Vector3f& box_sz){
    
    std::shared_ptr<fcl::CollisionGeometry<float>> ellipsoid(new fcl::Ellipsoid<float>(scale));
    std::shared_ptr<fcl::CollisionGeometry<float>> box2(new fcl::Box<float>(box_sz));

    fcl::Transform3f tf1 = fcl::Transform3f::Identity();
    tf1.translation() = ellipsoid_center;
    tf1.linear()=rotation;
    fcl::CollisionObjectf obj1(ellipsoid,tf1);

    fcl::Transform3f tf2 = fcl::Transform3f::Identity();
    tf2.translation() = box_center;
    fcl::CollisionObjectf obj2(box2,tf2);

    fcl::CollisionRequestf request;
    fcl::CollisionResultf result;

    request.gjk_solver_type = fcl::GJKSolverType::GST_INDEP;//specify solver type with the default type is GST_LIBCCD

    fcl::collide(&obj1,&obj2,request,result);

    //std::cout<<"test1 collide result:"<<result.isCollision()<<std::endl;
    return result.isCollision();
    
    return 1;
}

int main(int argc,char **argv){
    test1();
}
*/
