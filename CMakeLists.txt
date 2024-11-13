cmake_minimum_required(VERSION 3.28)
project(DD2375)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(lib/)

add_executable(baseline matmul-base/main.cpp)
add_executable(blas matmul-blas/main.cpp)
target_link_libraries(blas "-framework Accelerate")
