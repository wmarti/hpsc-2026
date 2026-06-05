#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err__));                               \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t err__ = (call);                                           \
    if (err__ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,  \
                   static_cast<int>(err__));                                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int kTileM = 320;
constexpr int kTileN = 64;
constexpr int kBlockK = 16;
constexpr int kThreads = 320;

__global__ void init_float_kernel(float *ptr, int64_t count, int seed) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < count; i += stride) {
    uint32_t x = static_cast<uint32_t>(i) * 1664525u +
                 static_cast<uint32_t>(seed) * 1013904223u;
    ptr[i] = (static_cast<float>(x & 255u) - 127.5f) * (1.0f / 128.0f);
  }
}

__global__ void fp32_to_half_kernel(const float *in, half *out, int64_t count) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < count; i += stride) {
    out[i] = __float2half_rn(in[i]);
  }
}

__global__ __launch_bounds__(kThreads, 2)
void wmma_bshared_320x64_kernel(int dim_m, int dim_n, int dim_k,
                                const half *__restrict__ a,
                                const half *__restrict__ b,
                                float *__restrict__ c) {
  __shared__ __align__(128) half tile_b[kTileN * kBlockK];

  int block_m = blockIdx.x * kTileM;
  int block_n = blockIdx.y * kTileN;
  int warp_id = threadIdx.x >> 5;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      wmma::fill_fragment(acc[r][n], 0.0f);
    }
  }

  for (int kk = 0; kk < dim_k; kk += kBlockK) {
    for (int idx = threadIdx.x; idx < kTileN * kBlockK; idx += kThreads) {
      int col = idx / kBlockK;
      int inner_k = idx - col * kBlockK;
      tile_b[idx] = b[(block_n + col) * dim_k + kk + inner_k];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
        b_frag[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      wmma::load_matrix_sync(b_frag[n], &tile_b[n * 16 * kBlockK], kBlockK);
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
      int row = block_m + warp_id * 32 + r * 16;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>
          a_frag;
      wmma::load_matrix_sync(a_frag, &a[kk * dim_m + row], dim_m);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        wmma::mma_sync(acc[r][n], a_frag, b_frag[n], acc[r][n]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int row = block_m + warp_id * 32 + r * 16;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      int col = block_n + n * 16;
      wmma::store_matrix_sync(&c[col * dim_m + row], acc[r][n], dim_m,
                              wmma::mem_col_major);
    }
  }
}

__global__ void diff_reduce_kernel(const float *a, const float *b, double *out,
                                   int64_t count) {
  __shared__ double smem[256];
  int tid = threadIdx.x;
  int64_t gid = int64_t(blockIdx.x) * blockDim.x + tid;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  double sum = 0.0;

  for (int64_t i = gid; i < count; i += stride) {
    sum += fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
  }
  smem[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smem[tid] += smem[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = smem[0];
  }
}

template <typename Fn>
float time_cuda_operation(Fn &&fn, int iterations) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < iterations + 2; ++i) {
    if (i == 2) {
      CHECK_CUDA(cudaEventRecord(start));
    }
    fn();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / iterations;
}

double average_abs_error(const float *a, const float *b, int64_t count) {
  int blocks = 4096;
  double *partial = nullptr;
  CHECK_CUDA(cudaMalloc(&partial, blocks * sizeof(double)));
  diff_reduce_kernel<<<blocks, 256>>>(a, b, partial, count);
  CHECK_CUDA(cudaGetLastError());
  std::vector<double> host(blocks);
  CHECK_CUDA(cudaMemcpy(host.data(), partial, blocks * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(partial));

  double sum = 0.0;
  for (double value : host) {
    sum += value;
  }
  return sum / static_cast<double>(count);
}

double gflops_from_ms(int m, int n, int k, float ms) {
  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                 static_cast<double>(k);
  return flops / (static_cast<double>(ms) * 1.0e6);
}

int main(int argc, char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  int iterations = 10;

  if (argc > 1) m = std::atoi(argv[1]);
  if (argc > 2) k = std::atoi(argv[2]);
  if (argc > 3) n = std::atoi(argv[3]);
  if (argc > 4) iterations = std::atoi(argv[4]);

  if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
    std::fprintf(stderr, "usage: %s [m k n iterations]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if ((m % kTileM) != 0 || (n % kTileN) != 0 || (k % kBlockK) != 0) {
    std::fprintf(stderr,
                 "This sm_90 Tensor Core kernel requires m a multiple of %d, "
                 "n a multiple of %d, and k a multiple of %d. Got m=%d k=%d "
                 "n=%d.\n",
                 kTileM, kTileN, kBlockK, m, k, n);
    return EXIT_FAILURE;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  int64_t size_a = int64_t(m) * k;
  int64_t size_b = int64_t(k) * n;
  int64_t size_c = int64_t(m) * n;

  float *a32 = nullptr;
  float *b32 = nullptr;
  half *a16 = nullptr;
  half *b16 = nullptr;
  float *c_cublas32 = nullptr;
  float *c_cublas16 = nullptr;
  float *c_raw = nullptr;

  CHECK_CUDA(cudaMalloc(&a32, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b32, size_b * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a16, size_a * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&b16, size_b * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&c_cublas32, size_c * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&c_cublas16, size_c * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&c_raw, size_c * sizeof(float)));

  int init_blocks = 4096;
  init_float_kernel<<<init_blocks, 256>>>(a32, size_a, 1);
  init_float_kernel<<<init_blocks, 256>>>(b32, size_b, 2);
  fp32_to_half_kernel<<<init_blocks, 256>>>(a32, a16, size_a);
  fp32_to_half_kernel<<<init_blocks, 256>>>(b32, b16, size_b);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  float cublas32_ms = time_cuda_operation([&]() {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, a32, CUDA_R_32F, m, b32, CUDA_R_32F, k,
                              &beta, c_cublas32, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }, iterations);

  float cublas16_ms = time_cuda_operation([&]() {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, a16, CUDA_R_16F, m, b16, CUDA_R_16F, k,
                              &beta, c_cublas16, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }, iterations);

  dim3 grid(m / kTileM, n / kTileN);
  float raw_ms = time_cuda_operation([&]() {
    wmma_bshared_320x64_kernel<<<grid, kThreads>>>(m, n, k, a16, b16, c_raw);
    CHECK_CUDA(cudaGetLastError());
  }, iterations);

  double err_vs_half = average_abs_error(c_cublas16, c_raw, size_c);
  double err_vs_fast16f = average_abs_error(c_cublas32, c_raw, size_c);

  double cublas32_gflops = gflops_from_ms(m, n, k, cublas32_ms);
  double cublas16_gflops = gflops_from_ms(m, n, k, cublas16_ms);
  double raw_gflops = gflops_from_ms(m, n, k, raw_ms);

  std::printf("M=%d K=%d N=%d iterations=%d\n", m, k, n, iterations);
  std::printf("cuBLAS FP32 input FAST_16F: %.3f ms, %.2f Gflop/s\n",
              cublas32_ms, cublas32_gflops);
  std::printf("cuBLAS FP16 input tensor op: %.3f ms, %.2f Gflop/s\n",
              cublas16_ms, cublas16_gflops);
  std::printf("raw CUDA WMMA B-shared 320x64: %.3f ms, %.2f Gflop/s\n",
              raw_ms, raw_gflops);
  std::printf("raw / cuBLAS FP32-input speedup: %.3fx\n",
              cublas32_ms / raw_ms);
  std::printf("raw / cuBLAS FP16-input speedup: %.3fx\n",
              cublas16_ms / raw_ms);
  std::printf("avg abs error vs cuBLAS FP16 input: %.8e\n", err_vs_half);
  std::printf("avg abs error vs cuBLAS FP32 FAST_16F: %.8e\n",
              err_vs_fast16f);

  CHECK_CUDA(cudaFree(a32));
  CHECK_CUDA(cudaFree(b32));
  CHECK_CUDA(cudaFree(a16));
  CHECK_CUDA(cudaFree(b16));
  CHECK_CUDA(cudaFree(c_cublas32));
  CHECK_CUDA(cudaFree(c_cublas16));
  CHECK_CUDA(cudaFree(c_raw));
  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
