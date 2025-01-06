//=================================================================================================================================
// Joe Schutte
//=================================================================================================================================
#include <Eigen/Dense>
#include "SurfaceParameters.h"


#define ForceNoMips_ true
#define EnableEWA_ true

namespace Selas
{
    float3x3 eigenToFloat3x3(const Eigen::Matrix3f& eigenMatrix) {
        float3x3 result;
        result.r0 = float3(eigenMatrix(0, 0), eigenMatrix(0, 1), eigenMatrix(0, 2));
        result.r1 = float3(eigenMatrix(1, 0), eigenMatrix(1, 1), eigenMatrix(1, 2));
        result.r2 = float3(eigenMatrix(2, 0), eigenMatrix(2, 1), eigenMatrix(2, 2));
        return result;
    }

    float3 eigenToFloat3(const Eigen::Vector3f& eigenVector) {
        float3 result;
        result = float3(eigenVector(0), eigenVector(1), eigenVector(2));
        return result;
    }

    void from_json(const nlohmann::json& j, float3& vec) {
        j.at(0).get_to(vec.x);
        j.at(1).get_to(vec.y);
        j.at(2).get_to(vec.z);
    }

    void SurfaceParameters::createParam(const Material& material, Gaussian& gaussian){
        //Eigen::Matrix3f rot_inv=gaussian.orientation.inverse();
        //printf("startcreate\n");
        Eigen::Matrix3f rot_inv=Eigen::Matrix3f::Identity();
        this->worldToTangent=eigenToFloat3x3(rot_inv);
        this->baseColor=material.baseColor;
        this->transmittanceColor=material.transmittanceColor;
        this->sheen = material.sheen;
        this->sheenTint = material.sheenTint;
        this->clearcoat = material.clearcoat;
        this->clearcoatGloss = material.clearcoatGloss;
        this->metallic = material.metallic;
        this->specTrans = material.specTrans;
        this->diffTrans = material.diffTrans;
        this->flatness = material.flatness;
        
        this->anisotropic = material.anisotropic;
        this->relativeIOR = material.relativeIOR;
        this->specularTint = material.specularTint;
        this->roughness = material.roughness;
        this->scatterDistance = material.scatterDistance;
        this->ior = material.ior;
        //printf("media\n");
    }

    bool Material::LoadFromJson(const std::string& filePath)
    {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filePath << std::endl;
            return false;
        }

        // 解析 JSON 文件
        json j;
        try {
            file >> j;
        }
        catch (const json::parse_error& e) {
            std::cerr << "Error: Failed to parse JSON file: " << e.what() << std::endl;
            return false;
        }

        // 直接赋值各项参数

        baseColor = j["baseColor"].get<float3>();
        transmittanceColor = j["transmittanceColor"].get<float3>();
        sheen = j["sheen"].get<float>();
        sheenTint = j["sheenTint"].get<float>();
        clearcoat = j["clearcoat"].get<float>();
        clearcoatGloss = j["clearcoatGloss"].get<float>();
        
        metallic = j["metallic"].get<float>();
        specTrans = j["specTrans"].get<float>();
        diffTrans = j["diffTrans"].get<float>();
        flatness = j["flatness"].get<float>();
        anisotropic = j["anisotropic"].get<float>();
        relativeIOR = j["relativeIOR"].get<float>();
        specularTint = j["specularTint"].get<float>();
        roughness = j["roughness"].get<float>();
        scatterDistance = j["scatterDistance"].get<float>();
        printf("111\n");
        ior = j["ior"].get<float>();

        return true;
    }
    //=============================================================================================================================
    
}
