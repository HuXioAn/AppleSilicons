#include <vecLib/vDSP.h>
#include "../lib/testlib.h"

void dsp(unsigned int nu, const float* left, const float* right, float* out) {
    int n = static_cast<int>(nu);
    vDSP_mmul(left, 1, right, 1, out, 1, n, n, n);
}

int main() {
    test_suite(dsp);
}