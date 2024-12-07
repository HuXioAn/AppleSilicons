#include <vecLib/cblas.h>
#include "testlib.h"

void blas(unsigned int nu, [[maybe_unused]] unsigned int memory_length, const float* left, const float* right, float* out) {
    int n = static_cast<int>(nu);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, left, n, right, n, 0, out, n);
}

int main() {
    test_suite(blas);
}