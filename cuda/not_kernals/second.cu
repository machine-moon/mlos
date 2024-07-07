#include <iostream>
#include <unistd.h>
#include <sys/random.h>
#include <cuda_runtime.h>

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y) 
{
      /*
    Threads (2): ||
    Blocks: [ ]
    Grid:
      Y
    X [ || ] [ || ] [ || ]
      [ || ] [ || ] [ || ]
      [ || ] [ || ] [ || ]
    */

    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int distance = blockDim.x * gridDim.x;
    for (int i = threadIndex; i < n; i += distance) {
        x[i] = x[i] * 0.95; // Scale x[i] by 0.95
        y[i] = y[i] + x[i]; // Add the scaled value of x[i] to y[i]
    }
}

int main(void) {
    int N = 1<<20; // 2^20 = 1M elements

    float *c_x, *c_y; // Host pointers
    float *g_x, *g_y; // Device pointers
    
    // Allocate host memory
    c_x = (float*)malloc(N * sizeof(float));
    c_y = (float*)malloc(N * sizeof(float));
    
    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        unsigned int random_value;
        getrandom(&random_value, sizeof(random_value), 0);
        c_x[i] = (float)random_value / UINT32_MAX; // Random float between 0 and 1
        getrandom(&random_value, sizeof(random_value), 0);
        c_y[i] = (float)random_value / UINT32_MAX; // Random float between 0 and 1
    }

    // Allocate device memory
    cudaMalloc(&g_x, N * sizeof(float));
    cudaMalloc(&g_y, N * sizeof(float));
    cudaMalloc(&g_result, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(g_x, c_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, c_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, g_x, g_y, g_result);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(c_x, g_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_y, g_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors (after scaling and addition)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(c_y[i] - (c_x[i] * 0.95)));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(g_x);
    cudaFree(g_y);
    cudaFree(g_result);
    free(c_x);
    free(c_y);
    
    return 0;
}