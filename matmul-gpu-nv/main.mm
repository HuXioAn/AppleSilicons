#include "testlib.h"
#include "gpulib.h"

int main() {
    id<MTLDevice> device;
    id<MTLComputePipelineState> matMultiplyFunctionPSO;
    if (!init_custom_shaders(device, matMultiplyFunctionPSO, "mat_mul_simple1")) {
        return 1;
    }
    test_suite([device, matMultiplyFunctionPSO](unsigned int n, unsigned int memory_length, void* left, void* right, void* out) {
        run_shader_matmul(device, matMultiplyFunctionPSO, n, memory_length, left, right, out);
    }, "../../../../data/");
    [device release];
}
