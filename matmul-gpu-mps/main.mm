#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include "testlib.h"

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device != nullptr);
    test_suite([device](unsigned int n, unsigned int memory_length, void* left, void* right, void* out) {
        id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy: left length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy: right length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy: out length:memory_length options:MTLResourceStorageModeShared deallocator:nil];
        assert(bufA != nullptr);
        assert(bufB != nullptr);
        assert(bufC != nullptr);
        // Define the matrix for the shaders.
        MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:desc];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:desc];
        // Define the multiplication.
        MPSMatrixMultiplication *matrixMultiplication = [[MPSMatrixMultiplication alloc]
                initWithDevice:device resultRows:n resultColumns:n interiorColumns:n];
        // Prepare the pipeline.
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        assert(commandQueue != nullptr);
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        assert(commandBuffer != nullptr);
        [matrixMultiplication encodeToCommandBuffer:commandBuffer leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        // Execute and clean.
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }, "../../../../data/");
    [device release];
}
