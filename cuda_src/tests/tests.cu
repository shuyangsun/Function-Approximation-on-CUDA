/*
 ============================================================================
 Name        : tests.hpp
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#include "tests.hpp"
#include "../helper/helper.hpp"
#include "../math_kernels/math_kernels.hpp"

#include <iostream>
#include <iomanip>

void PrintFPODuration(unsigned int const function_idx, double const duration, size_t const num_loop);
void PrintSFUDuration(unsigned int const function_idx, double const duration, size_t const num_loop);

void TestFPOs(dim3 const grid_dim, dim3 const block_dim) {
  std::cout << "Started testing FPO performance..." << std::endl;

  constexpr size_t num_loop{100};

  for (size_t i{0}; i < num_loop; ++i) { FPO_1<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_2<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_3<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_4<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_5<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_6<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_7<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_8<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_9<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_10<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_11<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_12<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_13<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_14<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_15<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_16<<<grid_dim, block_dim>>>(); }

  std::cout << "FPO performance testing finished." << std::endl;
}

void TestSFUs(dim3 const grid_dim, dim3 const block_dim) {
  std::cout << "Started testing SFU performance..." << std::endl;

  constexpr size_t num_loop{100};

  for (size_t i{0}; i < num_loop; ++i) { SFU_1<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_2<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_3<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_4<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_5<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_6<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_7<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_8<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_9<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_10<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_11<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_12<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_13<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_14<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_15<<<grid_dim, block_dim>>>(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_16<<<grid_dim, block_dim>>>(); }

  std::cout << "SFU performance testing finished." << std::endl;
}

void PrintFPODuration(unsigned int const function_idx, double const duration, size_t const num_loop) {
  std::cout << std::setprecision(3) << "FPO_1" << function_idx << ": " <<
    duration / num_loop * 1000.0 << " ms." << std::endl;
}

void PrintSFUDuration(unsigned int const function_idx, double const duration, size_t const num_loop) {
  std::cout << std::setprecision(3) << "SFU_1" << function_idx << ": " <<
    duration / num_loop * 1000.0 << " ms." << std::endl;
}

