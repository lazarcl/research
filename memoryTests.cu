//#include "arithmeticTests.cu"

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
class GlobalMemTest : public ArithmeticTestBase<T> {
public:
  GlobalMemTest(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  GlobalMemTest(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  void kernelSetup(cudaDeviceProp deviceProp) {
    ArithmeticTestBase<T>::kernelSetup(deviceProp);
  }

  void runKernel() {
      globalMemKernel<T><<<this->numBlocks, this->blockSize>>>
                      (this->n, this->iterNum, this->d_x);
  }
};
