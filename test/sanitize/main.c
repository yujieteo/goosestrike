#include "strike.h"
#include <stdio.h>
int main(int argc, char **argv) {
  goose_init(argc, argv);
  setup(1, 8, 0.009, 180.0);
  advance(200, 1e-4);
  printf("%d %g\n", get_steps(), get_time());
  return 0;
}
