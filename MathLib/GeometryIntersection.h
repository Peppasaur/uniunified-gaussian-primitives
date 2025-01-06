#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "MathLib/FloatStructs.h"
#include "SystemLib/BasicTypes.h"

namespace Selas
{
    namespace Intersection
    {
        void RaySphereNearest(float3 o, float3 d, float3 center, float r, float3& p);
        bool RaySphere(float3 o, float3 d, float3 c, float r, float3& p);
        bool RayAABox(float3 origin, float3 direction, float3 minPoint, float3 maxPoint);
        bool SweptSphereSphere(float3 c00, float3 c01, float r0, float3 c10, float3 c11, float r1);
        bool RayQuad(float3 o, float3 d, float3 v00, float3 v10, float3 v01, float3 v11);
    }
}
