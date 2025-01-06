#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "Shading/Scattering.h"
#include "MathLib/FloatStructs.h"

namespace Selas
{
    class CSampler;
    struct SurfaceParameters;

    bool SampleBsdfFunction(CSampler* sampler, const SurfaceParameters& surface, float3 v, BsdfSample& sample);
    float3 EvaluateBsdf(const SurfaceParameters& surface, float3 v, float3 l, float& forwardPdf, float& reversePdf);
}