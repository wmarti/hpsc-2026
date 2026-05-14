#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fillBucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void scanBucket(int *bucket, int *offset, int range) {
  int sum = 0;
  for (int i=0; i<range; i++) {
    offset[i] = sum;
    sum += bucket[i];
  }
}

__global__ void fillKey(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=range) return;
  int j = offset[i];
  for (; bucket[i]>0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *gpu_key, *bucket, *offset;
  cudaMallocManaged(&gpu_key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  for (int i=0; i<n; i++) {
    gpu_key[i] = key[i];
  }
  int m = 256;
  cudaMemset(bucket, 0, range*sizeof(int));
  fillBucket<<<(n+m-1)/m,m>>>(gpu_key, bucket, n);
  scanBucket<<<1,1>>>(bucket, offset, range);
  fillKey<<<(range+m-1)/m,m>>>(gpu_key, bucket, offset, range);
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) {
    key[i] = gpu_key[i];
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(gpu_key);
  cudaFree(bucket);
  cudaFree(offset);
}
