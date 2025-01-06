
//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "Shading/Fresnel.h"

#include "MathLib/Trigonometric.h"
#include "MathLib/FloatFuncs.h"
#include "SystemLib/MinMax.h"

namespace Selas
{
    namespace Fresnel
    {
        // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/

        //=========================================================================================================================
        float3 Schlick(float3 r0, float radians)
        {
            float exponential = Math::Powf(1.0f - radians, 5.0f);
            return r0 + (float3(1.0f) - r0) * exponential;
        }

        //=========================================================================================================================
        float Schlick(float r0, float radians)
        {
            return Lerp(1.0f, Fresnel::SchlickWeight(radians), r0);
        }

        //=========================================================================================================================
        float SchlickWeight(float u)
        {
            float m = Saturate(1.0f - u);
            float m2 = m * m;
            return m * m2 * m2;
        }

        //=========================================================================================================================
        float SchlickDielectic(float cosThetaI, float relativeIor)
        {
            float r0 = SchlickR0FromRelativeIOR(relativeIor);
            return r0 + (1.0f - r0) * SchlickWeight(cosThetaI);
        }

        //=========================================================================================================================
        float Dielectric(float cosThetaI, float ni, float nt)
        {
            // Copied from PBRT. This function calculates the full Fresnel term for a dielectric material.
            // See Sebastion Legarde's link above for details.

            cosThetaI = Clamp<float>(cosThetaI, -1.0f, 1.0f);

            // Swap index of refraction if this is coming from inside the surface
            if(cosThetaI < 0.0f) {
                float temp = ni;
                ni = nt;
                nt = temp;

                cosThetaI = -cosThetaI;
            }

            float sinThetaI = Math::Sqrtf(Max(0.0f, 1.0f - cosThetaI * cosThetaI));
            float sinThetaT = ni / nt * sinThetaI;

            // Check for total internal reflection
            if(sinThetaT >= 1) {
                return 1;
            }

            float cosThetaT = Math::Sqrtf(Max(0.0f, 1.0f - sinThetaT * sinThetaT));

            float rParallel     = ((nt * cosThetaI) - (ni * cosThetaT)) / ((nt * cosThetaI) + (ni * cosThetaT));
            float rPerpendicuar = ((ni * cosThetaI) - (nt * cosThetaT)) / ((ni * cosThetaI) + (nt * cosThetaT));
            return (rParallel * rParallel + rPerpendicuar * rPerpendicuar) / 2;
        }

        //=========================================================================================================================
        float SchlickR0FromRelativeIOR(float eta)
        {
            // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
            return Math::Square(eta - 1.0f) / Math::Square(eta + 1.0f);
        }
    }
}
