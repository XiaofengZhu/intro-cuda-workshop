#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

void add(float *a, float *b, float *c, long num)
{
    for (long ii = 0; ii < num; ++ii) {
        c[ii] = a[ii] + b[ii];
    }
}

int main(void)
{
    std::clock_t start_time;
    double duration01;
    double duration02;

    const long ARR_SIZE = 500000000;
    const size_t ARR_BYTES =  ARR_SIZE*sizeof(float);

    // Clock start
    start_time = std::clock();

    // Allocate input arrays
    float h_a[ARR_SIZE];
    float h_b[ARR_SIZE];
    float h_c[ARR_SIZE];

    // Assign numbers to array elements
    for (long i=0;  i<ARR_SIZE; i++){
        h_a[i] = float(i);
        h_b[i] = float(i)*2.0;
    }

    // Clock stop 01
    duration01 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time until function call: "<< duration01 << "s" << std::endl;

    // Call function add
    add(h_a, h_b, h_c, ARR_SIZE);

    // Clock stop 02
    duration02 = ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Computing time after function call: "<< duration02 << "s" << std::endl;

    // Output results
    for(long ii=0; ii<10; ii++){
        std::cout<< h_c[ii]<< ", ";
    }    
    std::cout<< std::endl;

    return 0;
}
