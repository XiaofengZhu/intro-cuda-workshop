#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

__global__
void add(float *h_a, float *h_b, float *h_c, long num)
{
    int idx = threadIdx.x+ blockIdx.x* blockDim.x;
    if (idx < num) {
        h_c[idx] = h_a[idx] + h_b[idx];
    }
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    std::clock_t start_time;
    double duration01;
    double duration02;
    double duration03;

    const long ARR_SIZE = 50000000;

    // Clock start
    start_time = std::clock();
    
    // Declare and alloc array on host
    float *h_a;
    float *h_b;
    float *h_c;

    // Allocate unified memory and record the return value for error
    err = cudaMallocManaged(&h_a, ARR_SIZE*sizeof(float));

    // Check for error, stop if error exists
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    // Allocate unified memory
    cudaMallocManaged(&h_b, ARR_SIZE*sizeof(float));
    cudaMallocManaged(&h_c, ARR_SIZE*sizeof(float));

    // initialize input array 
    for (long i=0;  i<ARR_SIZE; i++){
        h_a[i] = float(i);
        h_b[i] = float(i)*2.0;
    }

    // Clock stop 01
    duration01 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time until Kernel call: "<< duration01 << "s" << std::endl;

    // Call kernel function
    const int threadPerBlock = 1024;
    const int numBlock = ARR_SIZE/threadPerBlock+1;
    add<<<numBlock, threadPerBlock>>>(h_a, h_b, h_c, ARR_SIZE);

    // Block until the device has completed all tasks
    cudaDeviceSynchronize();

    // Clock stop 02
    duration02 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time after Kernel call: "<< duration02 << "s" << std::endl;

    // Clock stop 03
    duration03 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time after memory copy: "<< duration03 << "s" << std::endl;

    // Output results
    for(long ii=0; ii<10; ii++){
        std::cout<< h_c[ii]<< ", ";
    }    
    std::cout<< std::endl;

    return 0;
}
