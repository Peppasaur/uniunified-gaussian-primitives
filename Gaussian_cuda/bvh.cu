#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t error = call;                                                 \
        if (error != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define OPTIX_CHECK(call)                                             \
    do {                                                              \
        OptixResult res = call;                                       \
        if (res != OPTIX_SUCCESS) {                                   \
            std::cerr << "OptiX Error: " << optixGetErrorString(res); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Triangle data structure
struct Triangle {
    float3 v0, v1, v2;
};

struct BVHBuildResult {
    OptixTraversableHandle traversableHandle;
    OptixDeviceContext context;
    CUdeviceptr d_outputBuffer;
};

BVHBuildResult buildBVH(CUdeviceptr d_triangles, size_t numTriangles) {
    // Initialize CUDA and OptiX
    CUDA_CHECK(cudaFree(0)); // Initialize CUDA context
    OPTIX_CHECK(optixInit());

    OptixDeviceContext context = nullptr;
    CUcontext cuCtx = 0; // Current CUDA context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, nullptr, &context));

    // Build input
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    uint32_t triangleInputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices = static_cast<uint32_t>(numTriangles * 3);
    buildInput.triangleArray.vertexBuffers = &d_triangles;

    buildInput.triangleArray.indexBuffer = 0; // No explicit indices
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;

    buildInput.triangleArray.flags = &triangleInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    // Acceleration structure options
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Temporary buffers
    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes));

    CUdeviceptr d_tempBuffer = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_tempBuffer, bufferSizes.tempSizeInBytes));

    CUdeviceptr d_outputBuffer = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_outputBuffer, bufferSizes.outputSizeInBytes));

    OptixTraversableHandle traversableHandle = 0;
    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // CUDA stream
        &accelOptions,
        &buildInput,
        1,
        d_tempBuffer,
        bufferSizes.tempSizeInBytes,
        d_outputBuffer,
        bufferSizes.outputSizeInBytes,
        &traversableHandle,
        nullptr,
        0
    ));

    // Clean up temporary buffer
    CUDA_CHECK(cudaFree((void*)d_tempBuffer));

    std::cout << "BVH successfully built!" << std::endl;

    // Return results to the caller
    BVHBuildResult result;
    result.traversableHandle = traversableHandle;
    result.context = context;
    result.d_outputBuffer = d_outputBuffer;
    return result;
}

__global__ void Create_Tri(float* d_positions,float* d_orientations, float* d_scales,int N,Triangle* d_triangles){
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N)return;
    float3 rx=make_float3(d_orientations[9*idx]*d_scales[3*idx],d_orientations[9*idx+3]*d_scales[3*idx],d_orientations[9*idx+6]*d_scales[3*idx]);
    float3 ry=make_float3(d_orientations[9*idx+1]*d_scales[3*idx+1],d_orientations[9*idx+4]*d_scales[3*idx+1],d_orientations[9*idx+7]*d_scales[3*idx+1]);

    d_triangles[2*idx].v0.x=d_positions[3*idx]+rx.x; d_triangles[2*idx].v0.y=d_positions[3*idx+1]+rx.y; d_triangles[2*idx].v0.z=d_positions[3*idx+2]+rx.z;
    d_triangles[2*idx].v1.x=d_positions[3*idx]-rx.x; d_triangles[2*idx].v1.y=d_positions[3*idx+1]-rx.y; d_triangles[2*idx].v1.z=d_positions[3*idx+2]-rx.z;
    d_triangles[2*idx].v2.x=d_positions[3*idx]+ry.x; d_triangles[2*idx].v2.y=d_positions[3*idx+1]+ry.y; d_triangles[2*idx].v2.z=d_positions[3*idx+2]+ry.z;

    d_triangles[2*idx+1].v0.x=d_positions[3*idx]+rx.x; d_triangles[2*idx+1].v0.y=d_positions[3*idx+1]+rx.y; d_triangles[2*idx+1].v0.z=d_positions[3*idx+2]+rx.z;
    d_triangles[2*idx+1].v1.x=d_positions[3*idx]-rx.x; d_triangles[2*idx+1].v1.y=d_positions[3*idx+1]-rx.y; d_triangles[2*idx+1].v1.z=d_positions[3*idx+2]-rx.z;
    d_triangles[2*idx+1].v2.x=d_positions[3*idx]-ry.x; d_triangles[2*idx+1].v2.y=d_positions[3*idx+1]-ry.y; d_triangles[2*idx+1].v2.z=d_positions[3*idx+2]-ry.z;
    
}

void CreateScene(float* d_positions, float* d_orientations, float* d_scales,float* d_covs,float* magnitude,float* albedo,int N,float* rgbs,int res,float* env_rgbs,float* view_mat,float* space,float tanfovx,float tanfovy){
    CUdeviceptr d_triangles;
    CUDA_CHECK(cudaMalloc((void**)&d_triangles, 2 *N* sizeof(Triangle)));
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    Create_Tri<<<gridSize,blockSize>>>(d_positions,d_orientations,d_scales,N,(Triangle*)d_triangles);
    BVHBuildResult bvhResult=buildBVH(d_triangles,2*N);
    renderScene(d_positions, d_orientations, d_scales,d_covs,magnitude,albedo,N,bvhResult.traversableHandle,15,4,res,1,rgbs,env_rgbs,view_mat,tanfovx,tanfovy);
    
    CUDA_CHECK(cudaFree((void*)d_triangles));
    CUDA_CHECK(cudaFree((void*)bvhResult.d_outputBuffer));
    OPTIX_CHECK(optixDeviceContextDestroy(bvhResult.context));
}

