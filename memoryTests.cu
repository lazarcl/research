#include "arithmeticTests.h"
#include <stdio.h>



//------------------ L1 CACHE KERNELS -----------
template <typename T>
__global__
void l1MemKernel1(int n, int iterateNum, const T *x, T *y) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ volatile int index[256];
  index[threadIdx.x] = thread;

  T tot = 0;
  T var;

  for (int i = 0; i < iterateNum; i++) {
    const T *loc = &x[index[threadIdx.x]];
    var = __ldg(loc+n);
    var += var;
    tot += var;
  }
  y[thread] = tot;
  return;
}

template <typename T>
__global__
void l1MemKernel2(int n, int iterateNum, const T *x, T *y) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ volatile int index[256];
  index[threadIdx.x] = thread;

  T tot = 0;
  T var;

  for (int i = 0; i < iterateNum; i++) {
    const T *loc = &x[index[threadIdx.x]];
    var = __ldg(loc+n);
    var += __ldg(loc);
    tot += var;
  }
  y[thread] = tot;
  return;
}

//------------------ L2 CACHE KERNEL -----------
template <typename T>
__global__
void l2MemReadKernel1(int n, int iterateNum, volatile T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  T val = 0;
  for (int i = 0; i < iterateNum; i++) {
    val += x[thread];
    val += x[thread];
    val += val;
  }

  x[thread] = val;
  return;

}

template <typename T>
__global__
void l2MemReadKernel2(int n, int iterateNum, volatile T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  T val = 0;
  for (int i = 0; i < iterateNum; i++) {
    val += x[thread];
    val += x[thread];
    val += x[thread];
  }

  x[thread] = val;
  return;
}



//------------------ GLOBAL CACHE KERNELS -----------
__host__ __device__ unsigned reverse(unsigned nbits, unsigned val) {
  //From the bit twiddling hacks page http://graphics.stanford.edu/~seander/bithacks.html
  //Assuming unsigned is a 32-bit type, the most straightforward way is to flip the 32-bit value, 
  // than shift the flipped value by 32-nbits
  val = (((val & 0xaaaaaaaa) >> 1) | ((val & 0x55555555) << 1));
  val = (((val & 0xcccccccc) >> 2) | ((val & 0x33333333) << 2));
  val = (((val & 0xf0f0f0f0) >> 4) | ((val & 0x0f0f0f0f) << 4));
  val = (((val & 0xff00ff00) >> 8) | ((val & 0x00ff00ff) << 8));
  val = ((val >> 16) | (val << 16));
  return val >> (32-nbits);
}

template <typename T>
__global__
void globalMemKernel1(unsigned log2nblocksX, unsigned memblocksize, T *C, T *A, T x) {
  unsigned memblockIdx = blockIdx.x ^ reverse(log2nblocksX, blockIdx.y);
  unsigned idx = memblockIdx * memblocksize + threadIdx.x;
  float sum = 0.0f;
  for (unsigned i = 0; i < memblocksize/blockDim.x; i++) {
    sum = A[idx] * x + sum;
    idx += blockDim.x;
  }
  C[blockIdx.x*blockDim.x+threadIdx.x] = sum;
}

template <typename T>
__global__
void globalMemKernel2(unsigned log2nblocksX, unsigned memblocksize, T *C, T *A, T *B) {
  unsigned memblockIdx = blockIdx.x ^ reverse(log2nblocksX, blockIdx.y);
  unsigned idx = memblockIdx * memblocksize + threadIdx.x;
  float sum = 0.0f;
  for (unsigned i = 0; i < memblocksize/blockDim.x; i++) {
    sum = A[idx] * B[idx] + sum;
    idx += blockDim.x;
  }
  C[blockIdx.x*blockDim.x+threadIdx.x] = sum;
}


//------------------ SHARED MEMORY KERNEL -----------
template <typename T>
__global__
void sharedMemReadKernel1(int n, int iterateNum, volatile T *x) {
  //extern __shared__ volatile T s[]; //volatile to prevent optimization
  __shared__ volatile T s[1024];
  int thread = blockIdx.x*blockDim.x + threadIdx.x;

  s[threadIdx.x] = x[thread];

  volatile T val = 0;
  for (int i = 0; i < iterateNum; i++) {
    val += s[threadIdx.x];
    val += s[threadIdx.x];
    val += s[threadIdx.x];
    val += val;
  }

  x[thread] = val;

  return; 
}

template <typename T>
__global__
void sharedMemReadKernel2(int n, int iterateNum, volatile T *x) {
  //extern __shared__ volatile T s[];
  __shared__ volatile T s[1024];
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  
  s[threadIdx.x] = x[thread];

  volatile T val = 0;
  for (int i = 0; i < iterateNum; i++) {
    val += s[threadIdx.x];
    val += s[threadIdx.x];
    val += s[threadIdx.x];
    val += s[threadIdx.x];
  }

  x[thread] = val;

  return; 
}


//------------------ INITIALIZE ARRAY FOR KERNEL -----------
template <typename T>
__global__
void createData(int n, T *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // T a = 1.0;
  if (i < n) {
    //x[i] = i;
    x[i] = 0.0f;
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
    printf("n: %d\n", n);
    CUDA_ERROR( cudaMalloc(&d_x, 2*n*sizeof(T)) ); 
    createData<T><<<numBlocks, blockSize>>>(2*n, d_x);
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

//----------------------------------------------------------------
//---------------------- MEMORY TEST IMPLEMENTATIONS -------------
//----------------------------------------------------------------


//---------------------- L1 CACHE TESTING CLASSES -------------
template <typename T>
class L1MemTest1 : public MemoryTestBase<T> {
public:

  T *d_y;

  L1MemTest1(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 1;}
  L1MemTest1(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 1;}

  //should call base destructor after executing this destructor
  ~L1MemTest1() { 
    this->CUDA_ERROR(cudaFree(d_y));
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    MemoryTestBase<T>::kernelSetup(deviceProp);
    this->CUDA_ERROR( cudaMalloc(&d_y, this->n*sizeof(T)) ); 
    createData<T><<<this->numBlocks, this->blockSize>>>(this->n, d_y);
    printf("numblocks %d, blockSize %d, n %d, iterNum %d\n",this->numBlocks, this->blockSize, this->n, this->iterNum);
  }

  void runKernel() {
      l1MemKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x, d_y);
  }
};

template <typename T>
class L1MemTest2 : public MemoryTestBase<T> {
public:

  T *d_y;

  L1MemTest2(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 2;}
  L1MemTest2(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 2;}

  //should call base destructor after executing this destructor
  ~L1MemTest2() { 
    this->CUDA_ERROR(cudaFree(d_y));
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    MemoryTestBase<T>::kernelSetup(deviceProp);
    this->CUDA_ERROR( cudaMalloc(&d_y, this->n*sizeof(T)) ); 
    createData<T><<<this->numBlocks, this->blockSize>>>(this->n, d_y);
    printf("numblocks %d, blockSize %d, n %d, iterNum %d\n",this->numBlocks, this->blockSize, this->n, this->iterNum);
  }

  void runKernel() {
      l1MemKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x, d_y);
  }
};


//---------------------- L2 CACHE TESTING CLASSES -------------
template <typename T>
class L2MemReadTest1 : public MemoryTestBase<T> {
public:
  L2MemReadTest1(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 2;}
  L2MemReadTest1(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 2;}

  void runKernel() {
      l2MemReadKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class L2MemReadTest2 : public MemoryTestBase<T> {
public:
  L2MemReadTest2(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 3;}
  L2MemReadTest2(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 3;}

  void runKernel() {
      l2MemReadKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//---------------------- GLOBAL MEMORY TESTING CLASSES -------------
template <typename T>
class GlobalMemTest1 : public MemoryTestBase<T> {
public:
  unsigned memblocksize;
  unsigned log2nblocksX;
  T *A;
  T *B;
  T *C;
  T x = 0.0f; //constant passed into control kernel

  GlobalMemTest1(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 2;}
  GlobalMemTest1(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 2;}

  void kernelSetup(cudaDeviceProp deviceProp) {
    MemoryTestBase<T>::kernelSetup(deviceProp);

    size_t memcapacity = deviceProp.totalGlobalMem;
    unsigned memblockelems = this->blockSize*this->iterNum;
    memblocksize = memblockelems * sizeof(T);
    unsigned nblocksX = memcapacity / (4*memblocksize);
    //Round number of blocks down to next power of two
    log2nblocksX = 1;
    while ( (nblocksX >> (log2nblocksX+1)) > 0 ) {
      log2nblocksX++;
    }
    nblocksX = 1 << log2nblocksX;
    size_t in_buffer_size = nblocksX*memblocksize;

    T *A;
    this->CUDA_ERROR(cudaMalloc(&A, in_buffer_size));
    this->CUDA_ERROR(cudaMemset(A, 0, in_buffer_size));
   
    T *B;
    this->CUDA_ERROR(cudaMalloc(&B, in_buffer_size));
    this->CUDA_ERROR(cudaMemset(B, 0, in_buffer_size));

    T *C;
    this->CUDA_ERROR(cudaMalloc(&C, this->blockSize * nblocksX * sizeof(T)));
    this->CUDA_ERROR(cudaMemset(C, 0, this->blockSize * nblocksX * sizeof(T)));

  }

  void runKernel() {
      globalMemKernel1<T><<<this->numBlocks, this->blockSize>>>(log2nblocksX, memblocksize, C, A, x);
  }
};

template <typename T>
class GlobalMemTest2 : public MemoryTestBase<T> {
public:
  unsigned memblocksize;
  unsigned log2nblocksX;
  T *A;
  T *B;
  T *C;

  GlobalMemTest2(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 4;}
  GlobalMemTest2(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 4;}

  void kernelSetup(cudaDeviceProp deviceProp) {
    MemoryTestBase<T>::kernelSetup(deviceProp);

    size_t memcapacity = deviceProp.totalGlobalMem;
    unsigned memblockelems = this->blockSize*this->iterNum;
    memblocksize = memblockelems * sizeof(T);
    unsigned nblocksX = memcapacity / (4*memblocksize);
    //Round number of blocks down to next power of two
    log2nblocksX = 1;
    while ( (nblocksX >> (log2nblocksX+1)) > 0 ) {
      log2nblocksX++;
    }
    nblocksX = 1 << log2nblocksX;
    size_t in_buffer_size = nblocksX*memblocksize;

    T *A;
    this->CUDA_ERROR(cudaMalloc(&A, in_buffer_size));
    this->CUDA_ERROR(cudaMemset(A, 0, in_buffer_size));
   
    T *B;
    this->CUDA_ERROR(cudaMalloc(&B, in_buffer_size));
    this->CUDA_ERROR(cudaMemset(B, 0, in_buffer_size));

    T *C;
    this->CUDA_ERROR(cudaMalloc(&C, this->blockSize * nblocksX * sizeof(T)));
    this->CUDA_ERROR(cudaMemset(C, 0, this->blockSize * nblocksX * sizeof(T)));

  }

  void runKernel() {
      globalMemKernel2<T><<<this->numBlocks, this->blockSize>>>(log2nblocksX, memblocksize, C, A, B);
  }
};


//---------------------- SHARED MEMORY TESTING CLASSES -------------
template <typename T>
class SharedMemReadTest1 : public MemoryTestBase<T> {
public:

  unsigned int sharedMemRequest;

  SharedMemReadTest1(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 3;}
  SharedMemReadTest1(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 3;}

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp) {
    MemoryTestBase<T>::kernelSetup(deviceProp);
    sharedMemRequest = (unsigned int) (this->n * sizeof(T));
  }


  void runKernel() {
    sharedMemReadKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class SharedMemReadTest2 : public MemoryTestBase<T> {
public:

  unsigned int sharedMemRequest;

  SharedMemReadTest2(int blockSize, int iterNum) 
      : MemoryTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 4;}
  SharedMemReadTest2(int blockSize, int iterNum, int numBlockScale) 
      : MemoryTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 4;}

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp) {
    MemoryTestBase<T>::kernelSetup(deviceProp);
    sharedMemRequest = (unsigned int) (this->n * sizeof(T));
    //printf("  numBlocks %d, n %d \n", this->numBlocks, this->n);
    //printf("  sharedMemRequest %d\n", sharedMemRequest);
  }


  void runKernel() {
    sharedMemReadKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("  Error: %s\n", cudaGetErrorString(err));

  }
};
