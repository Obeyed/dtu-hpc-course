#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>


int main(int argc,char **argv)
{
    std::ofstream myfile;
    myfile.open ("seq_mapping.csv");

    // set these variables
    unsigned int times = 10;
    unsigned int IN_SIZE;
    unsigned int IN_BYTES;
    unsigned int OUT_SIZE;
    unsigned int OUT_BYTES;

    for (unsigned int rounds = 0; rounds<30; rounds++)
    {
        // Setting up variables
        IN_SIZE = 1<<rounds;
        IN_BYTES = sizeof(unsigned int)*IN_SIZE;
        OUT_SIZE = IN_SIZE;
        OUT_BYTES = IN_BYTES;
        printf("\ni = %d\n", rounds);
        printf("\n  ARRAY_SIZE = %d\n", IN_SIZE);
        printf("  ARRAY_BYTES = %d\n", IN_BYTES);

        // Setting host pointers
        unsigned int * h_in = (unsigned int*)malloc(IN_BYTES);
        unsigned int * h_out = (unsigned int*)malloc(OUT_BYTES);

        // Filling h_in
        for (unsigned int j = 0; j<IN_SIZE; j++) {h_in[j] = 1;}

        // setting up time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // running the code on the CPU $times times        
        for (unsigned int k = 0; k<times; k++)
        {
            for (unsigned int j = 0; j<OUT_SIZE; j++) {h_out[j] = h_in[j];}
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // calculating time
        float elapsedTime = .0f;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        elapsedTime = elapsedTime / ((float) times);
        printf(" time: %.5f\n", elapsedTime);
        free(h_in);
        free(h_out);
        myfile << elapsedTime << ",";
    }
    myfile.close();
    return 0;
}
