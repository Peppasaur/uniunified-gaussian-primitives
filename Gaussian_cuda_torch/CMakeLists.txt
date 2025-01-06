cmake_minimum_required(VERSION 3.8)
project(MyCUDAProject)
add_definitions(-DTORCH_USE_RTLD_GLOBAL)

message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")

# 启用 CUDA 支持
find_package(CUDA REQUIRED)

# 查找 pybind11
find_package(pybind11 REQUIRED PATHS /home/qinhaoran/.local/lib/python3.6/site-packages/pybind11/share/cmake/pybind11)

# 查找 cuDNN 库并设置路径
set(CUDNN_INCLUDE_PATH  /home/qinhaoran/.local/lib/python3.6/site-packages/nvidia/cudnn/include)
set(CUDNN_LIBRARY_PATH  /home/qinhaoran/.local/lib/python3.6/site-packages/nvidia/cudnn/lib/libcudnn.so.9)

# 查找 cuDNN 库

find_package(Torch REQUIRED PATHS /home/qinhaoran/.local/lib/python3.6/site-packages)

# 检查 cuDNN 库是否找到


# 添加 CUDA 和 C++ 源文件
set(SOURCES  gaussian_bbox.cu gaussian.cpp render_c.cu)

# 包含路径
include_directories(
    /home/qinhaoran/libs/include
    /home/qinhaoran/libs/include/eigen3
    /usr/local/include
    /home/qinhaoran/Selas/Source/Core/Gaussian_cuda
    ${TORCH_INCLUDE_DIRS}  # 包含 PyTorch 头文件
    ${CUDA_INCLUDE_DIRS}   # 包含 CUDA 头文件
    ${CUDNN_INCLUDE_DIR}    # 包含 cuDNN 头文件
)

# 链接目录
link_directories(
    /home/qinhaoran/libs/lib
    ${CUDNN_LIB_DIR}        # 链接 cuDNN 库目录
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)  # CUDA编译选项，可以根据需要添加更多选项

cuda_add_library(my_cuda_lib ${SOURCES})


# 创建 pybind11 扩展模块
pybind11_add_module(cuda_renderer port.cpp)
#add_executable(cuda_renderer ${SOURCES})


# 链接 PyTorch、CUDA 和 cuDNN 
target_link_libraries(cuda_renderer PUBLIC
    ${CUDA_LIBRARIES}   # 链接 CUDA 库
    ${CUDNN_LIBRARY}     # 链接 cuDNN 库
    ${TORCH_LIBRARIES}  # 链接 PyTorch 库
    my_cuda_lib
)

set(PYTHON_SITE_PACKAGES /home/qinhaoran/.local/lib/python3.6/site-packages)
install(TARGETS cuda_renderer DESTINATION ${PYTHON_SITE_PACKAGES})

