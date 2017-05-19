/*
 ============================================================================
 Name        : main.cu
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#include <iostream>
#include <iomanip>

#include "cuda_src/tests/tests.hpp"
#include "cuda_src/helper/helper.hpp"

int main(int arc, char *argv[]) {

  dim3 const grid_dim{1024};
  dim3 const block_dim{1024};
  TestFPOs(grid_dim, block_dim);
  TestSFUs(grid_dim, block_dim);

   TestMathKernels();
}

