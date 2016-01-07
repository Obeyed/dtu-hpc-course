#include <stdio.h>

int main(int argc,char **argv)
{
    const int ARRAY_SIZE = 1<<20;
    
    int acc = 0;
    int out[ARRAY_SIZE];
    int elements[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    for(int i = 1; i < ARRAY_SIZE; i++){
    	acc = acc + elements[i-1];
    	out[i] = acc;
    }

    for(int i = 0 ; i < ARRAY_SIZE; i++){
    	printf("%i ", out[i]);
    }

    return 0;
}
