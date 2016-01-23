#include <stdio.h>
#include <stddef.h>

int main(void) {
  size_t t = 5;
  int i = 5;

  printf("size_t: %lu\nint: %lu\n", sizeof(t), sizeof(i));

  return 0;
}
