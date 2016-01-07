#include <stdio.h>

/*
 * compile with `gcc -o test test.c
 * run `./test`
 */
int main(int argc, char **argv){
  const int A_SIZE = 1 << 21;
  const short INT_SIZE = sizeof(int);

  printf("Array Size: \t%d\nInt Size: \t%d\n", A_SIZE, INT_SIZE);

  return 0;
}
