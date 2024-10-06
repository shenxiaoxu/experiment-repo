#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 1000  // Size of the matrices

// CUDA kernel for matrix multiplication
__global__ void matrixMul(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float result = 0;
        for (int k = 0; k < n; ++k) {
            result += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = result;
    }
}

int main() {
    int size = N * N * sizeof(float);

    // Allocate host memory for matrices A, B, and C
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the matrix multiplication kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "CUDA Matrix Multiplication Time: " << duration.count() << " seconds\n";

    // Copy result matrix C back to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


// root@ip-10-2-153-167 ~]# ./cuda_matrix_multiplication
//   CUDA Matrix Multiplication Time: 0.137228 seconds