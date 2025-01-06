//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "MathLib/Projection.h"
#include "MathLib/FloatFuncs.h"
#include "MathLib/FloatStructs.h"
#include "MathLib/Trigonometric.h"
#include "SystemLib/JsAssert.h"
#include "SystemLib/MinMax.h"

#include <math.h>

namespace Selas
{
    namespace Math
    {

        // -- Cubemap projection assume d3d face ordering
        // D3DCUBEMAP_FACE_POSITIVE_X   = 0
        // D3DCUBEMAP_FACE_NEGATIVE_X   = 1
        // D3DCUBEMAP_FACE_POSITIVE_Y   = 2
        // D3DCUBEMAP_FACE_NEGATIVE_Y   = 3
        // D3DCUBEMAP_FACE_POSITIVE_Z   = 4
        // D3DCUBEMAP_FACE_NEGATIVE_Z   = 5

        //=========================================================================================================================
        // inputs:
        // face index 0 - 5
        // u in [-1,1]
        // v in [-1,1]
        float3 CubemapToCartesian(uint face, float u, float v)
        {
            switch(face) {
            case 0: return Normalize(float3(1.0f, -v, -u)); // +x
            case 1: return Normalize(float3(-1.0f, -v, u)); // -x
            case 2: return Normalize(float3(u, 1.0f, v)); // +y
            case 3: return Normalize(float3(u, -1.0f, -v)); // -y
            case 4: return Normalize(float3(u, -v, 1.0f)); // +z
            case 5: return Normalize(float3(-u, -v, -1.0f)); // -z
            }

            AssertMsg_(false, "Invalid face index");
            return float3(0.0f, 0.0f, 0.0f);
        }

        //=========================================================================================================================
        float3 CartesianToCubemap(const float3& xyz)
        {
            float absx = Math::Absf(xyz.x);
            float absy = Math::Absf(xyz.y);
            float absz = Math::Absf(xyz.z);

            float3 result;

            if(absx > absy && absx > absz) {
                float3 vec = xyz * (1.0f / absx);

                if(xyz.x > 0.0f) {
                    // +x
                    result.x = 0;
                    result.y = -vec.z;
                    result.z = -vec.y;
                }
                else {
                    // -x
                    result.x = 1;
                    result.y = vec.z;
                    result.z = -vec.y;
                }
            }
            else if(absy > absz) {
                float3 vec = xyz * (1.0f / absy);

                if(xyz.y > 0.0f) {
                    // +y
                    result.x = 2;
                    result.y = vec.x;
                    result.z = vec.z;
                }
                else {
                    // -y
                    result.x = 3;
                    result.y = vec.x;
                    result.z = -vec.z;
                }
            }
            else {
                float3 vec = xyz * (1.0f / absz);

                if(xyz.z > 0.0f) {
                    // +z
                    result.x = 4;
                    result.y = vec.x;
                    result.z = -vec.y;
                }
                else {
                    // -z
                    result.x = 5;
                    result.y = -vec.x;
                    result.z = -vec.y;
                }
            }

            return result;
        }

        //=========================================================================================================================
        void NormalizedCartesianToSpherical(const float3& v, float& theta, float& phi)
        {
            theta = Math::Acosf(v.y);
            phi = Math::Atan2f(v.z, v.x);
        }

        //=========================================================================================================================
        float3 SphericalToCartesian(const float3& rthetaphi)
        {
            float sintheta = Math::Sinf(rthetaphi.y);
            float costheta = Math::Cosf(rthetaphi.y);
            float sinphi = Math::Sinf(rthetaphi.z);
            float cosphi = Math::Cosf(rthetaphi.z);

            float3 xyz;
            xyz.x = rthetaphi.x * sintheta * cosphi;
            xyz.y = rthetaphi.x * costheta;
            xyz.z = rthetaphi.x * sintheta * sinphi;

            return xyz;
        }

        //=========================================================================================================================
        float3 SphericalToCartesian(float theta, float phi)
        {
            float sintheta = Math::Sinf(theta);
            float costheta = Math::Cosf(theta);
            float sinphi = Math::Sinf(phi);
            float cosphi = Math::Cosf(phi);

            float3 xyz;
            xyz.x = sintheta * cosphi;
            xyz.y = costheta;
            xyz.z = sintheta * sinphi;

            return xyz;
        }

        // -- The following two functions are from AMD's CubeMapGen source
        //=========================================================================================================================
        static float AreaElement(float x, float y)
        {
            return Math::Atan2f(x * y, Math::Sqrtf(x * x + y * y + 1));
        }

        //=========================================================================================================================
        float CubemapTexelSolidAngle(float u, float v, uint size)
        {
            float oo_size = 1.0f / size;

            float x0 = u - oo_size;
            float y0 = v - oo_size;
            float x1 = u + oo_size;
            float y1 = v + oo_size;
            float solid_angle = AreaElement(x0, y0) - AreaElement(x0, y1) - AreaElement(x1, y0) + AreaElement(x1, y1);

            return solid_angle;
        }

        //=========================================================================================================================
        float SphericalTexelSolidAngle(float theta0, float theta1, float phi0, float phi1)
        {
            float dtheta = theta1 - theta0;

            float sinTheta = Math::Sinf(theta1);
            float dphi = phi1 - phi0;

            return dtheta * (dphi * sinTheta);
        }
    }
}