#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include "testlib.h"
#include "ShaderParams.h"
#include <filesystem>
#include <iostream>
#include <string>

void metal(unsigned int n, const float* left, const float* right, float* out) {

}

void dbg(MTL::Buffer* buf) {
    auto bufc = buf->contents();
    if (!bufc) {
        std::cerr << "Fuck" << std::endl;
    }
    auto* contents = static_cast<float*>(buf->contents());
    for (int i = 0; i < 10; ++i) {
        std::cout << (float) contents[i] << " ";
    }
    std::cout << std::endl;
}

int main() {

    unsigned int n = 150;
    unsigned int n2 = n * n;
    // Get the device.
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    // Load the custom shader library.
    std::filesystem::path metalLibPathRelative = "../Resources/gpu_shaders.metallib";
    std::filesystem::path metalLibPathAbsolute = std::filesystem::absolute(metalLibPathRelative);
    NS::URL* shaderLib = NS::URL::fileURLWithPath(NS::String::string(metalLibPathAbsolute.c_str(), NS::StringEncoding::UTF8StringEncoding));
    NS::Error *error;
    MTL::Library *library = device->newLibrary(shaderLib, &error);
    if (library == nullptr) {
        std::cerr << "Failed to load library." << std::endl;
        std::cerr << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }
    // Create the actual matrix multiplication shader.
    auto str = NS::String::string("mat_mul_simple1", NS::ASCIIStringEncoding);
    MTL::Function *matMultiplyFunction = library->newFunction(str);
    if (matMultiplyFunction == nullptr) {
        std::cerr << "Failed to find the matrix multiplication shader." << std::endl;
        return 1;
    }
    // Create the pipeline.
    MTL::ComputePipelineState* matMultiplyFunctionPSO = device->newComputePipelineState(matMultiplyFunction, &error);
    if (matMultiplyFunctionPSO == nullptr) {
        std::cerr << "Failed to create the PSO: " << error << std::endl;
        return 1;
    }
    // Diagnostics.
    NS::UInteger thread_execution_width = matMultiplyFunctionPSO->threadExecutionWidth();
    std::cerr << "FYI, the thread execution wdith is: " << thread_execution_width << std::endl;
    NS::UInteger max_total_threads_per_threadgroup = matMultiplyFunctionPSO->maxTotalThreadsPerThreadgroup();
    std::cerr << "FYI, the maximum allowed threads per threadgoup is: " << max_total_threads_per_threadgroup << std::endl;
    // Prepare the command queue.
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    if (commandQueue == nullptr) {
        std::cerr << "Failed to get the command queue." << std::endl;
        return 1;
    }
    // Allocate the memory that we need.
    MatMulParams params {n, n, n};
    constexpr size_t l = 150 * 150;
    float arr[l] = {0};
    for (int i = 0; i < l; ++i) {
        arr[i] = i * 1.1;
    }
    size_t arrBufSize = l * sizeof(float);
    // We have to copy the memory since we can't page-align the buffer.
    MTL::Buffer* bufA = device->newBuffer(arr, arrBufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufB = device->newBuffer(arr, arrBufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufC = device->newBuffer(arr, arrBufSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufMeta = device->newBuffer(&params,sizeof(MatMulParams), MTL::ResourceStorageModeShared);
    // Queue up the command.
    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    computeEncoder->setComputePipelineState(matMultiplyFunctionPSO);
    computeEncoder->setBuffer(bufA, 0, 0);
    computeEncoder->setBuffer(bufB, 0, 1);
    computeEncoder->setBuffer(bufC, 0, 2);
    computeEncoder->setBuffer(bufMeta, 0, 3);
    // Set up the thread groups automatically-sih.
    const int x_threads_per_group = 8;
    const int y_threads_per_group = 8;
    assert(x_threads_per_group == y_threads_per_group);
    // The number of thread groups (i.e., blocks) per grid.
    const unsigned int x_group_count = (n + x_threads_per_group - 1) / x_threads_per_group;
    const unsigned int y_group_count = (n + y_threads_per_group - 1) / y_threads_per_group;
    MTL::Size threadGroups = MTL::Size::Make(x_group_count, y_group_count, 1);
    MTL::Size threadsPerThreadGroup = MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1);
    computeEncoder->dispatchThreadgroups(threadGroups, threadsPerThreadGroup);
    computeEncoder->endEncoding();
    // Start the shader!
    commandBuffer->commit();
    // Shader is still running here. Put other code here if you like.
    commandBuffer->waitUntilCompleted();
    // Garbage collect.
    bufMeta->release();
    bufC->release();
    bufB->release();
    bufA->release();
    device->release();
}

//int main() {
//    test_suite(metal);
//}