/*
 ============================================================================
 Name        : tests.hpp
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#ifndef OPPC_CUDA_SRC_TESTS_TESTS_HPP_
#define OPPC_CUDA_SRC_TESTS_TESTS_HPP_

void TestFPOs(dim3 const grid_dim, dim3 const block_dim);
void TestSFUs(dim3 const grid_dim, dim3 const block_dim);
void TestMathKernels();

#endif

