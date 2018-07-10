#ifndef MEMORYTESTS_H
#define MEMORYTESTS_H

#include "arithmeticTests.h"

template <typename T>
__global__
void globalMemKernel(int n, int iterateNum, volatile T *x);


template <typename T>
class GlobalMemTest : public ArithmeticTestBase<T> {
public:
  GlobalMemTest(int blockSize, int iterNum);
  GlobalMemTest(int blockSize, int iterNum, int numBlockScale);
  void kernelSetup(cudaDeviceProp deviceProp);
  void runKernel();
};

#endif
