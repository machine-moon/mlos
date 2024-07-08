#include <stdio.h>

// CUDA Kernel for adding two numbers
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
    // Host copies of a, b, c
    int a, b, c;
    // Device copies of a, b, c
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    // Setup input values
    a = 2;
    b = 7;

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with one thread
    add<<<1,1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Result of %d + %d = %d\n", a, b, c);

    return 0;
}