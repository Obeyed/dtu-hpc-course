#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

unsigned int computeBin(unsigned int * h_in, unsigned int l, unsigned int OUT_SIZE){
    return h_in[l] % OUT_SIZE;
}

int main(int argc,char **argv)
{
    std::ofstream myfile;
    myfile.open ("seq_histogram.csv");

    // set these variables
    const unsigned int times = 10;
    //const unsigned int binCount = 10;

    for (unsigned int rounds = 0; rounds<30; rounds++)
    {
        const unsigned int IN_SIZE = 1<<29;
        const unsigned int IN_BYTES = sizeof(unsigned int)*IN_SIZE;
        const unsigned int OUT_SIZE = 1<<rounds;
        const unsigned int OUT_BYTES = sizeof(unsigned int)*OUT_SIZE;
        printf("\ni = %d\n", rounds);
        printf("\n  ARRAY_SIZE = %d\n", IN_SIZE);
        printf("  ARRAY_BYTES = %d\n", IN_BYTES);
        unsigned int * h_in = (unsigned int*)malloc(IN_BYTES);
        unsigned int * h_out = (unsigned int*)malloc(OUT_BYTES);
        for (unsigned int j = 0; j<IN_SIZE; j++) {h_in[j] = j;}
        // setting up time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // running the code on the CPU $times times        
        for (unsigned int k = 0; k<times; k++)
        {
            for (unsigned int j = 0; j<OUT_SIZE; j++) {h_out[j] = 0;}
            //printf(" times = %d\n", k);
            for (unsigned int l = 0; l < IN_SIZE; ++l)
            {
                h_out[computeBin(h_in, l, OUT_SIZE)]++;
            }
            for (unsigned int x = 0; x<OUT_SIZE; x++) {printf("  h_out[%d] = %d\n", x, h_out[x]);}
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
