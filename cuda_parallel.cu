#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

__global__
void add(float *d_a, float *d_b, float *d_c, long num)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

int main(void)
{
    std::clock_t start_time;
    double duration01;
    double duration02;
    double duration03;

    const long ARR_SIZE =  500000000;
    const size_t ARR_BYTES =  ARR_SIZE*sizeof(float);

    // Clock start
    start_time = std::clock();
    
    // Declare and alloc array on host
    float h_a[ARR_SIZE];
    float h_b[ARR_SIZE];
    float h_c[ARR_SIZE];

    // initialize input array 
    for (long i=0;  i<ARR_SIZE; i++){
        h_a[i] = float(i);
        h_b[i] = float(i)*2.0;
    }

    // Declare and alloc array on device
    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc(&d_a, ARR_BYTES);
    cudaMalloc(&d_b, ARR_BYTES);
    cudaMalloc(&d_c, ARR_BYTES);

    // Transfer to device
    cudaMemcpy(d_a, h_a, ARR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ARR_BYTES, cudaMemcpyHostToDevice);

    // Clock stop 01
    duration01 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time until Kernel call: "<< duration01 << "s" << std::endl;

    // Call kernel function
    const int threadPerBlock = 1024;
    const int numBlock = ARR_SIZE/threadPerBlock+1;
    add<<<numBlock, threadPerBlock>>>(d_a, d_b, d_c, ARR_SIZE);

    // Block until the device has completed all tasks
    cudaDeviceSynchronize();

    // Clock stop 02
    duration02 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time after Kernel call: "<< duration02 << "s" << std::endl;

    // Transfer results to host
    cudaMemcpy(h_c, d_c, ARR_BYTES, cudaMemcpyDeviceToHost);

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
