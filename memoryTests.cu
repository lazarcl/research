#include "arithmeticTests.h"
#include <stdio.h>


//------------------ L1 CACHE KERNEL -----------
template <typename T>
__global__
void l1MemKernel(int n, int iterateNum, const T *x, T *y) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  T tot = 0;
  for (int j = 0; j < iterateNum; j++) {
    for (int i = 0; i < n; i++) {
      tot += x[i];
    }
  }
  y[thread] = tot;
  return;//TODO
}

//------------------ L2 CACHE KERNEL -----------
template <typename T>
__global__
void l2MemKernel(int n, int iterateNum, T *x) {

  for (int i = 0; i < iterateNum; i++) {
    for (int k = 0; k < n; k++) {
        x[(n-1)-k] = x[k];
    }
  }

}



//------------------ GLOBAL CACHE KERNELS -----------
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


//------------------ SHARED MEMORY KERNEL -----------
template <typename T>
__global__
void sharedMemKernel(int n, int iterateNum, T *x) {
  extern __shared__ T s[];
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i = 0; i < iterateNum; i++) {
    s[i] += s[n-i -1];
  }

  x[thread] = s[thread];

  return; //TODO
}



//------------------ INITIALIZE ARRAY FOR KERNEL -----------
template <typename T>
__global__
void createData(int n, T *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // T a = 1.0;
  if (i < n) {
    x[i] = i;
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
class L1MemTest : public MemoryTestBase<T> {
public:

  T *d_y;

  L1MemTest(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 1;}
  L1MemTest(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 1;}

  void kernelSetup(cudaDeviceProp deviceProp) {
    ArithmeticTestBase<T>::kernelSetup(deviceProp);
    CUDA_ERROR( cudaMalloc(&d_y, this->n*sizeof(T)) ); 
    createData<T><<<this->numBlocks, this->blockSize>>>(this->n, d_y);
  }

  void runKernel() {
      l1MemKernel<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x, d_y);
  }
};


template <typename T>
class L2MemTest : public MemoryTestBase<T> {
public:
  L2MemTest(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 2;}
  L2MemTest(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 2;}

  void runKernel() {
      l2MemKernel<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class GlobalMemTest : public MemoryTestBase<T> {
public:
  GlobalMemTest(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 2;}
  GlobalMemTest(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 2;}

  void runKernel() {
      globalMemKernel<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class SharedMemTest : public MemoryTestBase<T> {
public:

  unsigned int sharedMemRequest;

  SharedMemTest(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 2;}
  SharedMemTest(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 2;}

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp) {
    ArithmeticTestBase<T>::kernelSetup(deviceProp);
    sharedMemRequest = (unsigned int) (this->n * sizeof(T));
  }


  void runKernel() {
      sharedMemKernel<T><<<this->numBlocks, this->blockSize, sharedMemRequest>>>(this->n, this->iterNum, this->d_x);
  }
};
