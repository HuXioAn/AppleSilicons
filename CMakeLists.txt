cmake_minimum_required(VERSION 3.28)
project(DD2375)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake")

include_directories(lib/)

add_executable(baseline matmul-cpu-base/main.cpp)

add_executable(blas matmul-blas/main.cpp)
target_link_libraries(blas "-framework Accelerate")

add_executable(dsp matmul-dsp/main.cpp)
target_link_libraries(dsp "-framework Accelerate")

include(MetalShaderSupport)
enable_language(Metal)
enable_language(OBJCXX)

add_metal_shader_library(gpu_shaders
        metal_shaders/mat_mul_simple1.metal
        metal_shaders/mat_mul_optimized_nv.metal
)
target_include_directories(gpu_shaders PRIVATE lib/)

add_executable(gpu_baseline MACOSX_BUNDLE matmul-gpu-base/main.cpp )
target_include_directories(gpu_baseline PRIVATE metal/)
target_link_libraries(gpu_baseline "-framework Metal -framework Foundation -framework QuartzCore -framework CoreGraphics")
target_embed_metal_shader_libraries(gpu_baseline gpu_shaders)

add_executable(gpu_mps MACOSX_BUNDLE matmul-gpu-mps/main.mm)
target_include_directories(gpu_mps PRIVATE metal/)
target_link_libraries(gpu_mps "-framework MetalPerformanceShaders -framework Metal -framework Foundation -framework QuartzCore")
target_embed_metal_shader_libraries(gpu_mps gpu_shaders)
