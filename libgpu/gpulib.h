#ifndef DD2375_GPULIB_H
#define DD2375_GPULIB_H

#include <Metal/Metal.h>
#include <string>
#include <shaderinfo.h>

bool init_custom_shaders(id<MTLDevice> &device, id<MTLComputePipelineState> &matMultiplyFunctionPSO, const std::string& shader) {
    device = MTLCreateSystemDefaultDevice();
    assert(device != nullptr);
    // Load the shaders.
    std::filesystem::path metalLibPathRelative = "../Resources/gpu_shaders.metallib";
    std::filesystem::path metalLibPathAbsolute = std::filesystem::absolute(metalLibPathRelative);
    NSString* metalLibPath = [[NSString alloc] initWithUTF8String:metalLibPathAbsolute.c_str()];
    NSURL* shaderLib = [NSURL fileURLWithPath:metalLibPath];
    NSError *error;
    id<MTLLibrary> library = [device newLibraryWithURL:shaderLib error:&error];
    if (library == nullptr) {
        std::cerr << "Failed to load library." << std::endl;
        std::cerr << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    // Create the functions.
    NSString* str = [[NSString alloc] initWithCString:shader.c_str()];
    id<MTLFunction> matMultiplyFunction = [library newFunctionWithName:str];
    assert(matMultiplyFunction != nullptr);
    matMultiplyFunctionPSO = [device newComputePipelineStateWithFunction:matMultiplyFunction error:&error];
    assert(matMultiplyFunctionPSO != nullptr);
    return true;
}

void run_shader_matmul(id<MTLDevice> device, id<MTLComputePipelineState> matMultiplyFunctionPSO, unsigned int n, unsigned int memory_length, void* left, void* right, void* out) {
    MatMulParams params {n, n, n};
    id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy: left length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy: right length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy: out length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bufMeta = [device newBufferWithBytes:&params length:sizeof(MatMulParams) options:MTLResourceStorageModeShared];
    assert(bufA != nullptr);
    assert(bufB != nullptr);
    assert(bufC != nullptr);
    assert(bufMeta != nullptr);
    // Prepare the pipeline.
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    assert(commandQueue != nullptr);
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    assert(commandBuffer != nullptr);
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    assert(commandEncoder != nullptr);
    [commandEncoder setComputePipelineState:matMultiplyFunctionPSO];
    [commandEncoder setBuffer:bufA offset:0 atIndex:0];
    [commandEncoder setBuffer:bufB offset:0 atIndex:1];
    [commandEncoder setBuffer:bufC offset:0 atIndex:2];
    [commandEncoder setBuffer:bufMeta offset:0 atIndex:3];
    // Set thread group sizing information.
    const int x_threads_per_group = 8;
    const int y_threads_per_group = 8;
    assert(x_threads_per_group == y_threads_per_group);
    // The number of thread groups (i.e., blocks) per grid.
    const unsigned int x_group_count = (n + x_threads_per_group - 1) / x_threads_per_group;
    const unsigned int y_group_count = (n + y_threads_per_group - 1) / y_threads_per_group;
    MTLSize threadGroups = MTLSizeMake(x_group_count, y_group_count, 1);
    MTLSize threadsPerThreadGroup = MTLSizeMake(x_threads_per_group, y_threads_per_group, 1);
    [commandEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadsPerThreadGroup];
    [commandEncoder endEncoding];
    // Execute and clean.
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

#endif
