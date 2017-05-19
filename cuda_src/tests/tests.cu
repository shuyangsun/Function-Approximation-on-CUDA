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
  float *d_out;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float)));

  constexpr size_t num_loop{100};
  double start_time{0.0};
  double end_time{0.0};
  double duration{0.0};

  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_1); } PrintFPODuration(1, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_2); } PrintFPODuration(2, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_3); } PrintFPODuration(3, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_4); } PrintFPODuration(4, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_5); } PrintFPODuration(5, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_6); } PrintFPODuration(6, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_7); } PrintFPODuration(7, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_8); } PrintFPODuration(8, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_9); } PrintFPODuration(9, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_10); } PrintFPODuration(10, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_11); } PrintFPODuration(11, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_12); } PrintFPODuration(12, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_13); } PrintFPODuration(13, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_14); } PrintFPODuration(14, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_15); } PrintFPODuration(15, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(FPO_16); } PrintFPODuration(16, duration, num_loop);

  CHECK_CUDA_ERR(cudaFree(d_out));
  std::cout << "FPO performance testing finished." << std::endl;
}

void TestSFUs(dim3 const grid_dim, dim3 const block_dim) {
  std::cout << "Started testing SFU performance..." << std::endl;
  float *d_out;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float)));

  constexpr size_t num_loop{100};
  double start_time{0.0};
  double end_time{0.0};
  double duration{0.0};

  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_1); } PrintSFUDuration(1, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_2); } PrintSFUDuration(2, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_3); } PrintSFUDuration(3, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_4); } PrintSFUDuration(4, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_5); } PrintSFUDuration(5, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_6); } PrintSFUDuration(6, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_7); } PrintSFUDuration(7, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_8); } PrintSFUDuration(8, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_9); } PrintSFUDuration(9, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_10); } PrintSFUDuration(10, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_11); } PrintSFUDuration(11, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_12); } PrintSFUDuration(12, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_13); } PrintSFUDuration(13, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_14); } PrintSFUDuration(14, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_15); } PrintSFUDuration(15, duration, num_loop);
  for (size_t i{0}; i < num_loop; ++i) { TIME_KERNEL_1(SFU_16); } PrintSFUDuration(16, duration, num_loop);

  CHECK_CUDA_ERR(cudaFree(d_out));
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

