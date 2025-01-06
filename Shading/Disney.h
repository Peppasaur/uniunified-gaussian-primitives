#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================

#include "MathLib/FloatStructs.h"
#include "Shading/SurfaceScattering.h"
#include "Shading/SurfaceParameters.h"
#include "Shading/IntegratorContexts.h"
#include "Gaussian/gaussian.h"

#include "Shading/Fresnel.h"
#include "Shading/Ggx.h"
#include "MathLib/FloatFuncs.h"
#include "MathLib/Trigonometric.h"
#include "MathLib/Projection.h"
#include "SystemLib/MinMax.h"

namespace Selas
{
    class CSampler;
    struct HitParameters;
    struct SurfaceParameters;
    struct BsdfSample;

    // -- BSDF evaluation for next event estimation
    float3 EvaluateDisney(const SurfaceParameters& surface, float3 v, float3 l, bool thin, float& forwardPdf, float& reversePdf);

    // -- Shaders
    bool SampleDisney(CSampler* sampler, const SurfaceParameters& surface, float3 v, bool thin, BsdfSample& sample);
}