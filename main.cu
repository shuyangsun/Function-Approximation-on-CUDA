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
#include <cstdlib>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CHECK_CUDA_ERR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

double CPUSecond();
float RandomFloat();

enum class KernelFunc {
  Polynomial, Trigonometry
};

double TimeKernel(const KernelFunc func, const float * const data_h, size_t const data_size, unsigned int const block_dim_x);

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size);
__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size);

int main(int arc, char *argv[]) {

  // Customization for testing.
  size_t const num_loop{10};
  float const max_gig_count{4.0f};
  float const step_size{0.5f};

  // Initialize attributes
  size_t const max_data_size{static_cast<size_t>(max_gig_count * 1024 * 1024 * 1024)};
  size_t const max_num_ele{max_data_size / sizeof(float)};

  // Generate random data array
  float * const data_h{reinterpret_cast<float*>(malloc(max_data_size))};
  std::cout << std::setprecision(2) << "Generating random float array of " << max_gig_count << "GB..." << std::endl;
  srand(time(NULL));
  for (size_t i{0}; i < max_num_ele; ++i) {
    data_h[i] = RandomFloat();
  }
  std::cout << "Finished generating random float array." << std::endl;

  // Start outter loop (data size loop)
  for (float i{step_size}; i <= max_gig_count; i += step_size) {
    std::cout << std::setprecision(2) << "------------ " << i << "GB ------------" << std::endl;
    double duration_trig{0.0};
    double duration_poly{0.0};

    // Start inner loop (repetition loop)
    for (size_t j{0}; j < num_loop; ++j) {

      float const gig_count{i};
      size_t const data_size{static_cast<size_t>(gig_count * (1 << 30))};
      unsigned int const block_dim_x{1024};

      // Trig Kernel
      double const tmp_dur_trig{TimeKernel(KernelFunc::Trigonometry, data_h, data_size, block_dim_x)};
      duration_trig += tmp_dur_trig * 1000.0;

      // Poly Kernel
      double const tmp_dur_poly{TimeKernel(KernelFunc::Polynomial, data_h, data_size, block_dim_x)};
      duration_poly += tmp_dur_poly * 1000.0;
    }

    // Calculate average
    duration_trig /= num_loop;
    duration_poly /= num_loop;

    // Print out information
    std::cout << std::setprecision(3) << "Finished trig kernel in average " << duration_trig << " ms." << std::endl;
    std::cout << std::setprecision(3) << "Finished poly kernel in average " << duration_poly << " ms." << std::endl;
    std::cout << std::setprecision(5) << "Trig time / Poly time: " << duration_trig/duration_poly << std::endl;
    std::cout << std::setprecision(3) << "Speed up: " << (1.0f - duration_poly/duration_trig) * 100 << "%" << std::endl;
  }

  free(data_h);
  CHECK_CUDA_ERR(cudaDeviceReset());

  std::cout << "-------------------------------" << std::endl;
  return 0;
}

double TimeKernel(const KernelFunc func, const float * const data_h, size_t const data_size, unsigned int const block_dim_x) {
  float *data_d;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&data_d), data_size));
  CHECK_CUDA_ERR(cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice));

  float *res;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&res), data_size/2));

  size_t const num_ele{data_size / sizeof(float)};
  dim3 const block_dim{block_dim_x};
  dim3 const grid_dim{static_cast<unsigned int>((num_ele + block_dim.x - 1) / block_dim.x)};

  double start{0.0f};
  double end{0.0f};

  switch (func) {
    case KernelFunc::Trigonometry:
      start = CPUSecond();
      TrigFunc<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      end = CPUSecond();
      break;
    case KernelFunc::Polynomial:
      start = CPUSecond();
      PolyFunc<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      end = CPUSecond();
      break;
    default:
      break;
  };

  CHECK_CUDA_ERR(cudaFree(data_d));
  CHECK_CUDA_ERR(cudaFree(res));

  return end - start;
}

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size) {

  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float a{data_in[idx_2]};
    const float b{data_in[idx_2 + 1]};
    // const float res1{0.36f + 0.68f * a * (1 + 0.28f * a * (1 + 0.78f * a * (1 - 0.57f * a * (1 + 0.68f * a * (1 + 0.68f * a * (1 + 0.68f * a * (1 + 0.68f * a)))))))};
    // const float res2{0.36f + 0.68f * b * (1 + 0.28f * b * (1 + 0.78f * b * (1 - 0.57f * b * (1 + 0.68f * b * (1 + 0.68f * b * (1 + 0.68f * b * (1 + 0.68f * b)))))))};
    const float res1{0.34f * (a - 0.5f) * (a - 0.65f) * (a - 0.2f) * (a + 0.5f) * (a - 1.2f) * (a - 0.4f) * (a - 1.5f) * (a - 2.5f)};
    const float res2{0.34f * (b - 0.5f) * (b - 0.65f) * (b - 0.2f) * (b + 0.5f) * (b - 1.2f) * (b - 0.4f) * (b - 1.5f) * (b - 2.5f)};

    data_out[idx] = res2 - res1;
  }
}

__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};
  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};
    const float res1{-0.75f * (x1 * x1 - 2) * __cosf(x1) + 0.01f * (9 * x1 * x1 - 2) * __cosf(3 * x1) + 1.5f * x1 * __sinf(x1) + 2.1f * __sinf(x1) - 0.052f * x1 * __sinf(3 * x1) + 0.25f * (3 * x1)};
    const float res2{-0.75f * (x2 * x2 - 2) * __cosf(x2) + 0.01f * (9 * x2 * x2 - 2) * __cosf(3 * x2) + 1.5f * x2 * __sinf(x2) + 2.1f * __sinf(x2) - 0.052f * x2 * __sinf(3 * x2) + 0.25f * (3 * x2)};

    data_out[idx] = res2 - res1;
  }
}

float RandomFloat() {
  return ((float)rand())/((float)rand());
}

double CPUSecond() {
  return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
  if (err == cudaSuccess)
    return;
  std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
  exit (1);
}

