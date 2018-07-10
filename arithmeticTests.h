#ifndef ARITHMETICTESTS_H
#define ARITHMETICTESTS_H
//mult, add and FMA Kernels work for FP32, FP64, and Int 
//  without register limiting
//#include <stdio.h>


//------------ EXPERIMENT VOLATILE KERNEL FOR BASEPOWER 2 ---------
template <typename T>
__global__
void addKernel1Volatile(int n, int iterateNum, T *x);

template <typename T>
__global__
void multKernel1_nonVolitile(int n, int iterateNum, T *x);


//------------ BASEPOW1: SET SHARED MEMORY KERNEL ---------
template <typename T>
__global__
void addKernel1_DynamicSharedMem(int n, int iterateNum, T *x);

template <typename T>
__global__
void multKernel_DynamicSharedMem(int n, int iterateNum, T *x);


template <typename T>
__global__
void fmaKernel_DynamicSharedMem(int n, int iterateNum, T *x);


//------------ ADDITION KERNELS ---------
template <typename T>
__global__
void addKernel1(int n, int iterateNum, T *x);


template <typename T>
__global__
void addKernel2(int n, int iterateNum, T *x);


//------------ MULTIPLICATION KERNELS ---------
template <typename T>
__global__
void multKernel1(int n, int iterateNum, T *x);


template <typename T>
__global__
void multKernel2(int n, int iterateNum, T *x);


//------------ FMA KERNELS ---------
template <typename T>
__global__
void fmaKernel1(int n, int iterateNum, T *x);

template <typename T>
__global__
void fmaKernel2(int n, int iterateNum, T *x);



template <typename T>
__global__
void createData(int n, T *x);


//------------ BASE CLASS FOR TESTS TO INHERIT FROM ---------
template <typename T>
class ArithmeticTestBase {
public: 

  T *d_x;
  int n;
  int iterNum;
  int numBlocks;
  int blockSize;
  int numBlockScale;
  int opsPerIteration; //number of operations in one iteration. Not including loop calculations

  ArithmeticTestBase(int blockSize, int iterNum);
  ArithmeticTestBase(int blockSize, int iterNum, int numBlockScale);

  ~ArithmeticTestBase();

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

//------------ TEST CLASS FOR BASE POWER MEASUREMENT APPR 1 ---------
template <typename T>
class AddKernel1TestSetSharedMem : public ArithmeticTestBase<T> {
public:

  unsigned int sharedMemRequest;
  float sharedMemScale; 

  AddKernel1TestSetSharedMem(int blockSize, int iterNum);
  AddKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale);

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp);
  void setSharedMem(float newScale);

  void runKernel();
};

template <typename T>
class MultKernel1TestSetSharedMem : public ArithmeticTestBase<T> {
public:

  unsigned int sharedMemRequest;
  float sharedMemScale; 

  MultKernel1TestSetSharedMem(int blockSize, int iterNum);
  MultKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale);

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp);

  void setSharedMem(float newScale);

  void runKernel();
};

template <typename T>
class FMAKernel1TestSetSharedMem : public ArithmeticTestBase<T> {
public:

  unsigned int sharedMemRequest;
  float sharedMemScale; 

  FMAKernel1TestSetSharedMem(int blockSize, int iterNum);
  FMAKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale);

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp);
  void setSharedMem(float newScale);
  void runKernel();
};

//------------ TEST CASE FOR BASE POWER APPR 2 ---------
template <typename T>
class AddKernel1TestVolatile : public ArithmeticTestBase<T> {
public:
  AddKernel1TestVolatile(int blockSize, int iterNum);
  AddKernel1TestVolatile(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};

template <typename T>
class MultKernel1TestNonVolatile : public ArithmeticTestBase<T> {
public:
  MultKernel1TestNonVolatile(int blockSize, int iterNum);;
  MultKernel1TestNonVolatile(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};


//------------ ADDITION TEST CLASSES ---------
template <typename T>
class AddKernel1Test : public ArithmeticTestBase<T> {
public:
  //this->opsPerIteration = 6;
  AddKernel1Test(int blockSize, int iterNum);
  AddKernel1Test(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};

template <typename T>
class AddKernel2Test : public ArithmeticTestBase<T> {
public:
  AddKernel2Test(int blockSize, int iterNum);
  AddKernel2Test(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};


//------------ MULTIPLICATION TEST CLASSES ---------
template <typename T>
class MultKernel1Test : public ArithmeticTestBase<T> {
public:
  MultKernel1Test(int blockSize, int iterNum);
  MultKernel1Test(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};

template <typename T>
class MultKernel2Test : public ArithmeticTestBase<T> {
public:
  MultKernel2Test(int blockSize, int iterNum);
  MultKernel2Test(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};


//------------ FMA TEST CLASSES ---------
template <typename T>
class FmaKernel1Test : public ArithmeticTestBase<T> {
public:
  FmaKernel1Test(int blockSize, int iterNum);
  FmaKernel1Test(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};

template <typename T>
class FmaKernel2Test : public ArithmeticTestBase<T> {
public:
  FmaKernel2Test(int blockSize, int iterNum);
  FmaKernel2Test(int blockSize, int iterNum, int numBlockScale);

  void runKernel();
};



#endif




