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
  float const max_gig_count{2.0f};
  float const step_size{1.0f};

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

__device__ float PolyRes(float const val) {
  return 0.18449222732258544f + 0.27022193536223005f * val * (1.0f - 0.3865057722380988f * val * (1.0f + 0.5148217493442544f * val * (1.0f + 0.5168314642472727f * val)));
}

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size) {

  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyRes(x1)};
    const float res2{PolyRes(x2)};
    const float res3{0.2f + 0.27022193536223005f * x1 * (1.0f - 0.3865057722380988f * x1 * (1.0f + 0.5148217493442544f * x1 * (1.0f + 0.5168314642472727f * x1)))};
    const float res4{0.2f + 0.27022193536223005f * x2 * (1.0f - 0.3865057722380988f * x2 * (1.0f + 0.5148217493442544f * x2 * (1.0f + 0.5168314642472727f * x2)))};
    const float res5{1.2f + 0.37022193536223005f * x1 * (1.0f - 0.3865057722380988f * x1 * (1.0f + 0.5148217493442544f * x1 * (1.0f + 0.5168314642472727f * x1)))};
    const float res6{1.2f + 0.37022193536223005f * x2 * (1.0f - 0.3865057722380988f * x2 * (1.0f + 0.5148217493442544f * x2 * (1.0f + 0.5168314642472727f * x2)))};
    const float res7{1.1f + 0.77022193536223005f * x1 * (1.0f - 0.3865057722380988f * x1 * (1.0f + 0.5148217493442544f * x1 * (1.0f + 0.5168314642472727f * x1)))};
    const float res8{1.1f + 0.77022193536223005f * x2 * (1.0f - 0.3865057722380988f * x2 * (1.0f + 0.5148217493442544f * x2 * (1.0f + 0.5168314642472727f * x2)))};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
  }
}

__device__ float TrigRes(float const val) {
  return ((0.75f * val * val - 4.71239f * val + 5.9022f) * __cosf(val) + (-0.0833333f * val * val + 0.523599f * val - 0.803949f) * __cosf(3.0f * val) + 4.5f * val - 1.5f * val * __sinf(val) + 0.0555556f * val * __sinf(3.0f * val) + 6.96239f * __sinf(val) + 0.0754671f * __sinf(3.0f * val))/(9.0f * 3.141592653f);
}

__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};
  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{TrigRes(x1)};
    const float res2{TrigRes(x2)};
    const float res3{((0.8f * x1 * x1 - 1.71239f * x1 + 5.9022f) * __cosf(x1) + (-0.0833333f * x1 * x1 + 0.523599f * x1 - 0.803949f) * __cosf(3.0f * x1) + 4.5f * x1 - 1.5f * x1 * __sinf(x1) + 0.0555556f * x1 * __sinf(3.0f * x1) + 6.96239f * __sinf(x1) + 0.0754671f * __sinf(3.0f * x1))/(9.0f * 3.141592653f)};
    const float res4{((0.8f * x2 * x2 - 1.71239f * x2 + 5.9022f) * __cosf(x2) + (-0.0833333f * x2 * x2 + 0.523599f * x2 - 0.803949f) * __cosf(3.0f * x2) + 4.5f * x2 - 1.5f * x2 * __sinf(x2) + 0.0555556f * x2 * __sinf(3.0f * x2) + 6.96239f * __sinf(x2) + 0.0754671f * __sinf(3.0f * x2))/(9.0f * 3.141592653f)};
    const float res5{((1.8f * x1 * x1 - 2.71239f * x1 + 5.9022f) * __cosf(x1) + (-0.0833333f * x1 * x1 + 0.523599f * x1 - 0.803949f) * __cosf(3.0f * x1) + 4.5f * x1 - 1.5f * x1 * __sinf(x1) + 0.0555556f * x1 * __sinf(3.0f * x1) + 6.96239f * __sinf(x1) + 0.0754671f * __sinf(3.0f * x1))/(9.0f * 3.141592653f)};
    const float res6{((1.8f * x2 * x2 - 2.71239f * x2 + 5.9022f) * __cosf(x2) + (-0.0833333f * x2 * x2 + 0.523599f * x2 - 0.803949f) * __cosf(3.0f * x2) + 4.5f * x2 - 1.5f * x2 * __sinf(x2) + 0.0555556f * x2 * __sinf(3.0f * x2) + 6.96239f * __sinf(x2) + 0.0754671f * __sinf(3.0f * x2))/(9.0f * 3.141592653f)};
    const float res7{((1.8f * x1 * x1 - 3.71239f * x1 + 5.9022f) * __cosf(x1) + (-0.0833333f * x1 * x1 + 0.523599f * x1 - 0.803949f) * __cosf(3.0f * x1) + 4.5f * x1 - 1.5f * x1 * __sinf(x1) + 0.0555556f * x1 * __sinf(3.0f * x1) + 6.96239f * __sinf(x1) + 0.0754671f * __sinf(3.0f * x1))/(9.0f * 3.141592653f)};
    const float res8{((1.8f * x2 * x2 - 3.71239f * x2 + 5.9022f) * __cosf(x2) + (-0.0833333f * x2 * x2 + 0.523599f * x2 - 0.803949f) * __cosf(3.0f * x2) + 4.5f * x2 - 1.5f * x2 * __sinf(x2) + 0.0555556f * x2 * __sinf(3.0f * x2) + 6.96239f * __sinf(x2) + 0.0754671f * __sinf(3.0f * x2))/(9.0f * 3.141592653f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
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

