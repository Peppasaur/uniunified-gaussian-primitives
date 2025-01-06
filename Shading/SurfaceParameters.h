#pragma once

//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "MathLib/FloatStructs.h"
#include "Gaussian/gaussian.h"

using json = nlohmann::json;

namespace Selas
{
    struct GIIntegratorContext;
    struct HitParameters;
    struct ModelResource;
    struct MaterialResourceData;

    struct Material
    {
        
        float3 baseColor;
        float3 transmittanceColor;
        float sheen;
        float sheenTint;
        float clearcoat;
        float clearcoatGloss;
        float metallic;
        float specTrans;
        float diffTrans;
        float flatness;
        float anisotropic;
        float relativeIOR;
        float specularTint;
        float roughness;
        float scatterDistance;

        float ior;
        
        bool LoadFromJson(const std::string& filePath);
    };

    struct SurfaceParameters
    {
        //float3 position;
        float3x3 worldToTangent;
        //float error;

        //float3 view;
        
        float3 baseColor;
        float3 transmittanceColor;
        float sheen;
        float sheenTint;
        float clearcoat;
        float clearcoatGloss;
        float metallic;
        float specTrans;
        float diffTrans;
        float flatness;
        float anisotropic;
        float relativeIOR;
        float specularTint;
        float roughness;
        float scatterDistance;

        float ior;
        
        void createParam(const Selas::Material& material, Gaussian& gaussian);
        // -- material layer info
        //ShaderType shader;
        //uint32 materialFlags;

        //uint32 lightSetIndex;
    };

    


}