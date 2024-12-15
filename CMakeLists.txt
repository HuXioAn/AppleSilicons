cmake_minimum_required(VERSION 3.28)
project(DD2375)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake")

if (DEFINED MATRIX_N)
    # Pass it to the compiler as a preprocessor definition
    add_definitions(-DMATRIX_N=${MATRIX_N})
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif ()

include_directories(lib/)

set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
set(OpenMP_CXX_FLAG "-Xclang -fopenmp")
set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
set(OpenMP_CXX_LIB_NAMES "libomp")
set(OpenMP_C_FLAG "-Xclang -fopenmp")
set(OpenMP_C_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
set(OpenMP_C_LIB_NAMES "libomp")
set(OpenMP_libomp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")

find_package(OpenMP COMPONENTS CXX REQUIRED)

add_executable(baseline matmul-cpu-base/main.cpp)

add_executable(omp matmul-cpu-omp/main.cpp)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX)

add_executable(blas matmul-blas/main.cpp)
target_link_libraries(blas PRIVATE "-framework Accelerate")

add_executable(dsp matmul-dsp/main.cpp)
target_link_libraries(dsp PRIVATE "-framework Accelerate")

include(MetalShaderSupport)
enable_language(Metal)
enable_language(OBJCXX)

add_metal_shader_library(gpu_shaders
        metal_shaders/mat_mul_simple1.metal
        metal_shaders/mat_mul_optimized_nv.metal
)
target_include_directories(gpu_shaders PRIVATE lib/)
target_include_directories(gpu_shaders PRIVATE libgpu/)

add_executable(gpu_baseline MACOSX_BUNDLE matmul-gpu-base/main.mm)
target_include_directories(gpu_baseline PRIVATE libgpu/)
target_link_libraries(gpu_baseline "-framework Metal -framework Foundation -framework QuartzCore -framework CoreGraphics")
target_embed_metal_shader_libraries(gpu_baseline gpu_shaders)

add_executable(gpu_mps MACOSX_BUNDLE matmul-gpu-mps/main.mm)
target_link_libraries(gpu_mps "-framework MetalPerformanceShaders -framework Metal -framework Foundation -framework QuartzCore")
target_embed_metal_shader_libraries(gpu_mps gpu_shaders)

add_executable(gpu_nv MACOSX_BUNDLE matmul-gpu-nv/main.mm)
target_include_directories(gpu_nv PRIVATE libgpu/)
target_link_libraries(gpu_nv "-framework Metal -framework Foundation -framework QuartzCore")
target_embed_metal_shader_libraries(gpu_nv gpu_shaders)