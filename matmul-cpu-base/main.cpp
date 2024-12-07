#include "testlib.h"

void naive(unsigned int n, [[maybe_unused]] unsigned int memory_length, const float* left, const float* right, float* out) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                out[INDEX(n, i, j)] += left[INDEX(n, i, k)] * right[INDEX(n, k, j)];
            }
        }
    }
}

int main() {
    test_suite(naive);
}