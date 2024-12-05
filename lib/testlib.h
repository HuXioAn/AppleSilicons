#ifndef testlib
#define testlib

#include <chrono>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>

#define INDEX(N,R,C) (N*R + C)

void internal_load_matrix(const std::string& pathPrefix, unsigned short n, unsigned short idx, float* arr) {
    auto fileName = pathPrefix + "matrix-" + std::to_string(n) + "-" + std::to_string(idx) + ".float32";
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "could not open '" << fileName << "'" << std::endl;
        return;
    }
    if (!file.read(reinterpret_cast<char*>(arr), static_cast<std::streamsize>(n * n * sizeof(float)))) {
        // Successfully read a float
        std::cerr << "error reading '" << fileName << "' as float32" << std::endl;
        return;
    }
}

template<typename MatMul>
void test_suite(MatMul consumer, const std::string &pathPrefix = "../data/") {
    // { 150u, 500u, 1000u, 2500u, 5000u, 10000u }
    for (unsigned int n : { 150u }) {
        auto n2 = n * n;
        // Create all the arrays.
        float left[n2];
        float right[n2];
        float out[n2];
        // Load them.
        internal_load_matrix(pathPrefix, n, 0, left);
        internal_load_matrix(pathPrefix, n, 1, right);
        // Perform matrix multiplication.
        auto before = std::chrono::high_resolution_clock::now();
        consumer(n, left, right, out);
        auto after = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
        std::cout << n << ": check " << out[INDEX(n, 0, 0)] << " took " << elapsed.count() << "ns" << std::endl;
    }
}


#endif
