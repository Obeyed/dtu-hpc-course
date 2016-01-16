#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

int main(int argc,char **argv)
{
    std::ofstream myfile;
    myfile.open ("seq_reduce.csv");
    const unsigned int times = 10;
    for (unsigned int i = 0; i<30; i++)
    {
        const unsigned int IN_SIZE = 1<<i;
        const unsigned int IN_BYTES = sizeof(unsigned int)*IN_SIZE;
        const unsigned int OUT_SIZE = 1;
        const unsigned int OUT_BYTES = sizeof(unsigned int)*OUT_SIZE;
        printf("\ni = %d\n", i);
        printf("\n  ARRAY_SIZE = %d\n", IN_SIZE);
        printf("  ARRAY_BYTES = %d\n", IN_BYTES);
        unsigned int * h_in = (unsigned int*)malloc(IN_BYTES);
        unsigned int * h_out = (unsigned int*)malloc(OUT_BYTES);
        for (unsigned int j = 0; j<IN_SIZE; j++) {h_in[j] = 1;}

        // setting up time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // running the code on the CPU $times times        
        for (unsigned int k = 0; k<times; k++)
        {
            h_out[0] = 0;
            for (unsigned int l = 0; l < IN_SIZE; ++l)
            {
                h_out[0] += h_in[l];
            }
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // calculating time
        float elapsedTime = .0f;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        elapsedTime = elapsedTime / ((float) times);
        printf(" time: %.5f\n", elapsedTime);

        myfile << elapsedTime << ",";
    }
    myfile.close();
    return 0;
}
