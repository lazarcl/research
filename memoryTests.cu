 #include "arithmeticTests.h"
#include <stdio.h>

template <typename T>
__global__
void globalMemKernel(int n, int iterateNum, volatile T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  volatile T a = 0;

  for (int i = 0; i < iterateNum; i++) {
    for (int j = 0; j < n; j++) {
      a = x[j];
    }
  }
  x[thread] = a;
}

template <typename T>
__global__
void createData(int n, T *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  T a = 1.0;
  if (i < n) {
    x[i] = a;
  }
}

template <typename T>
class MemoryTestBase {
public: 

  T *d_x;
  int n;
  int iterNum;
  int numBlocks;
  int blockSize;
  int numBlockScale;
  int opsPerIteration; //number of operations in one iteration. Not including loop calculations

  MemoryTestBase(int blockSize, int iterNum)
    : iterNum(iterNum), blockSize(blockSize), numBlockScale(360)
  { opsPerIteration = 0;}
  MemoryTestBase(int blockSize, int iterNum, int numBlockScale)
    : iterNum(iterNum), blockSize(blockSize), numBlockScale(numBlockScale)
  { opsPerIteration = 0;}

  ~MemoryTestBase() {
    CUDA_ERROR( cudaFree(d_x) );
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    numBlocks = deviceProp.multiProcessorCount * numBlockScale;
    n = numBlocks * blockSize;
    CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(T)) ); 
    createData<T><<<numBlocks, blockSize>>>(n, d_x);
  }

  //get the number of threads launched in the kernel. Must be 
  //called after kernelSetup() or the neccisary fields may not be initialized
  int getNumThreads() {
    return numBlocks * blockSize;
  }

  //return the number of operations that are executed in the kernel's loop
  //for the specified number of operations.
  //Ex: 6 operations per iteration * 1000000 iterations = 6000000 operations
  int getOpsPerThread() {
    return opsPerIteration * iterNum;
  }

  void runKernel();

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda error in test class: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};


template <typename T>
class GlobalMemTest : public MemoryTestBase<T> {
public:
  GlobalMemTest(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  GlobalMemTest(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  void runKernel() {
      globalMemKernel<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};
