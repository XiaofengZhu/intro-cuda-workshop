#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum_shared_mem(float *array)
{
    int idx = threadIdx.x;
    float sum=0.0f;

    // Share among threads within the same block
    __shared__ float sh_array[1024];
    sh_array[idx] = array[idx];

    // Syncronize threads within the same block   
    __syncthreads();

    for (int i=0; i<=idx; i++){
        sum+= sh_array[i];
    }

    __syncthreads();

    array[idx] = sum;
}


__global__ void sum_global_mem(float *array)
{
    int idx = threadIdx.x;
    float sum=0.0f;

    for (int i=0; i<=idx; i++){
        sum+= array[i];
    }
    __syncthreads();

    array[idx] = sum;
}


int main(void)
{
    std::clock_t start_time;
    double duration01;
    double duration02;
    double duration03;    

    const int ARR_BYTES =  1024*sizeof(float);

    // Clock start
    start_time = std::clock();
    
    // Declare and alloc array on host
    float h_array[1024];

    // initialize input array 
    for (int i=0;  i<1024; i++){
        h_array[i] = float(i);
    }

    // Declare and alloc array on device
    float *d_array;
    cudaMalloc(&d_array, ARR_BYTES);

    // Transfer to device
    cudaMemcpy(d_array, h_array, ARR_BYTES, cudaMemcpyHostToDevice);

    // Clock stop 01
    duration01 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time before Kernel call: "<< duration01 << "s" << std::endl;

    // Call kernel function with shared memory
    sum_shared_mem<<<1, 1024>>>(d_array);

    // Call kernel function with shared memory
//    sum_global_mem<<<1, 1024>>>(d_array);   

    // Clock stop 02
    duration02 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time after Kernel call: "<< duration02 << "s" << std::endl;

    // Transfer results to host
    cudaMemcpy(h_array, d_array, ARR_BYTES, cudaMemcpyDeviceToHost);

    // Clock stop 03
    duration03 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time after memory copy: "<< duration03 << "s" << std::endl;

    // Output results
    for(int ii=0; ii<10; ii++){
        std::cout<< h_array[ii]<< ", ";
    }    
    std::cout<< std::endl;

    return 0;
}
