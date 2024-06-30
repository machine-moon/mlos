#include <iostream>
#include <unistd.h>
#include <sys/random.h>
#include <cuda_runtime.h>

// CUDA kernel to add elements of two arrays
// the actual kernal
__global__ void add(int n, float *x, float *y) {

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
    int i;
    for (i = threadIndex; i < n; i += distance)
        x[i] = x[i] * 0.95;
        y[i] = y[i] + x[i];
}

int main(void) {
    int N = 1<<20; // bitwise opr, 2^20 = 1M elements
    // float *x, *y; 
    // Allocate unified memory – accessible from CPU or GPU
    // cudaMallocManaged(&x, N*sizeof(float)); // x->array[2M]
    // cudaMallocManaged(&y, N*sizeof(float));



    // better
    float *c_x, *c_y; // cpu pointers
    float *g_x, *g_y; // gpu pointers
    
    // Allocate host memory
    h_x = (float*)malloc(N * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));
    
    // Initialize x and y arrays on the host with random numbers between 0 and 1
    for (int i = 0; i < N; i++) {
        unsigned int random_value;
        getrandom(&random_value, sizeof(random_value), 0);
        h_x[i] = (float)random_value / UINT32_MAX;
        getrandom(&random_value, sizeof(random_value), 0);
        h_y[i] = (float)random_value / UINT32_MAX;
    }

    // Allocate device memory
    cudaMalloc(&g_x, N * sizeof(float));
    cudaMalloc(&g_y, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(g_x, c_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, c_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (values will not be exactly 3.0f due to random initialization)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(g_y[i]-(c_x[i] + g_x[i])));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}

