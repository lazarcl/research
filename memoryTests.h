#ifndef MEMORYTESTS_H
#define MEMORYTESTS_H

//#include "arithmeticTests.h"

template <typename T>
__global__
void globalMemKernel(int n, int iterateNum, volatile T *x);


template <typename T>
__global__
void createData(int n, T *x);


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

  MemoryTestBase(int blockSize, int iterNum);
  MemoryTestBase(int blockSize, int iterNum, int numBlockScale);

  ~MemoryTestBase();

  void kernelSetup(cudaDeviceProp deviceProp);

  //get the number of threads launched in the kernel. Must be 
  //called after kernelSetup() or the neccisary fields may not be initialized
  int getNumThreads();

  //return the number of operations that are executed in the kernel's loop
  //for the specified number of operations.
  //Ex: 6 operations per iteration * 1000000 iterations = 6000000 operations
  int getOpsPerThread();

  void runKernel();

  void CUDA_ERROR(cudaError_t e);

};


template <typename T>
class GlobalMemTest : public MemoryTestBase<T> {
public:
  GlobalMemTest(int blockSize, int iterNum);
  GlobalMemTest(int blockSize, int iterNum, int numBlockScale);
 // void kernelSetup(cudaDeviceProp deviceProp);
  void runKernel();
};

#endif
