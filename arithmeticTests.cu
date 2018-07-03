
//mult, add and FMA Kernels work for FP32, FP64, and Int 
//  without register limiting


//------------ EXPERIMENT VOLATILE KERNEL FOR BASEPOWER 2 ---------
template <typename T>
__global__
void addKernel1Volatile(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // volatile T b = 2;
  // volatile T c = 2;
  volatile T b = 1;
  volatile T c = 1;
  for (int i = 0; i < iterateNum; i++) {
    b = a + b;
    c = a + b;
    a = c + a;
    c = b + a;
    b = c + a;
    a = b + c;
  }
  x[thread] = a;
}

template <typename T>
__global__
void multKernel1_nonVolitile(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 1, c = 1;
  // T b = 0, c = 0;
  //T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a * b;
    c = a * b;
    a = c * a;
    c = b * a;
    b = c * a;
    a = b * c;
  }
  x[thread] = a;
}


//------------ BASEPOW1: SET SHARED MEMORY KERNEL ---------
template <typename T>
__global__
void addKernel1_DynamicSharedMem(int n, int iterateNum, T *x) {
  extern __shared__ int s[];
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 1, c = 1;
  // T b = 0, c = 0;
  //T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + b;
    c = a + b;
    a = c + a;
    c = b + a;
    b = c + a;
    a = b + c;
  }
  x[thread] = a;
}

template <typename T>
__global__
void multKernel_DynamicSharedMem(int n, int iterateNum, T *x) {
  extern __shared__ int s[];

  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 1, c = 1;
  // T b = 0, c = 0;
  //T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a * b;
    c = a * b;
    a = c * a;
    c = b * a;
    b = c * a;
    a = b * c;
    a = a/1000000 +1;
    b = b/1000000 +1;
    c = c/1000000 +1;
  }
  x[thread] = a;
}


template <typename T>
__global__
void fmaKernel_DynamicSharedMem(int n, int iterateNum, T *x) {
  extern __shared__ int s[];
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  // T b = 1, c = 1;
  T b = 0.25, c = 0.25, d = 0.1875;
  // T b = 0, c = 0;
  for (int i = 0; i < iterateNum; i++) {
    // b = a*b + c;
    // c = a*b + a;
    // a = c*b + a;
    // c = b*a + c;
    // b = a*b + b;
    // a = b*c + a;
    // c = b*a + c;
    // b = a*b + b;
    // a = b*c + a;
    b = a*b + d;
    c = a*b + d;
    a = c*b + d;
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;
  }
  x[thread] = a;
}


//------------ ADDITION KERNELS ---------
template <typename T>
__global__
void addKernel1(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 1, c = 1;
  // T b = 0, c = 0;
  //T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + b;
    c = a + b;
    a = c + a;
    c = b + a;
    b = c + a;
    a = b + c;
  }
  x[thread] = a;
}


template <typename T>
__global__
void addKernel2(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 1, c = 1;
  // T b = 0, c = 0;
  //T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + b;
    c = a + b;
    a = c + a;
    c = b + a;
    b = c + a;
    a = b + c;
    c = b + a;
    b = c + a;
    a = b + c;
  }
  x[thread] = a;
}


//------------ MULTIPLICATION KERNELS ---------
template <typename T>
__global__
void multKernel1(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 2, c = 2;
  // T b = 0, c = 0;
  // T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a * b;
    c = a * b;
    a = c * a;
    c = b * a;
    b = c * a;
    a = b * c;
    a = (a >> 10) + 2;
    b = (b >> 10) + 2;
    c = (c >> 10) + 2;
  }
  x[thread] = a;
}


template <typename T>
__global__
void multKernel2(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  // T b = 2.22507e-308, c = 2.22507e-308;
  // T b = 0, c = 0;
  T b = 2, c = 2;
  // T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a * b;
    c = a * b;
    a = c * a;
    c = b * a;
    b = c * a;
    a = b * c;
    c = b * a;
    b = c * a;
    a = b * c;
    a = (a >> 10) + 2;
    b = (b >> 10) + 2;
    c = (c >> 10) + 2;
  }
  x[thread] = a;
}


//------------ FMA KERNELS ---------
template <typename T>
__global__
void fmaKernel1(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  //float b = 1.175494351e+38f, c = 1.175494351e+38f;
  //float b=1f, c=1f;
  // T b = 2.22507e-308, c = 2.22507e-308;
  T b = 0.25, c = 0.25, d = 0.1875;

  // T b = 0, c = 0;
  for (int i = 0; i < iterateNum; i++) {
    // b = a*b + c;
    // c = a*b + a;
    // a = c*b + a;
    // c = b*a + c;
    // b = a*b + b;
    // a = b*c + a;
    // c = b*a + c;
    // b = a*b + b;
    // a = b*c + a;
    b = a*b + d;
    c = a*b + d;
    a = c*b + d;
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;

  }
  x[thread] = a;
}

template <typename T>
__global__
void fmaKernel2(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  //float b = 1.175494351e+38f, c = 1.175494351e+38f;
  //T b = 2.22507e-308, c = 2.22507e-308;
  T b = 0.25, c = 0.25, d = 0.1875;
  // T b = 0, c = 0;
  for (int i = 0; i < iterateNum; i++) {
    b = a*b + d;
    c = a*b + d;
    a = c*b + d;
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;
    
    c = b*a + d;
    b = a*c + d;
    a = b*c + d;    
  }
  x[thread] = a;
}



template <typename T>
__global__
void createData(int n, T *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  T a = 1.175494351e-38f;
//T a = 1.0e+38f;
//  T a = 1.0;
  // T a = 0;
  // T a = 2.22507e-308;
  T a = 0.25;
  if (i < n) {
    x[i] = a;
  }
}


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

  ArithmeticTestBase(int blockSize, int iterNum)
    : iterNum(iterNum), blockSize(blockSize), numBlockScale(360)
  { opsPerIteration = 0;}
  ArithmeticTestBase(int blockSize, int iterNum, int numBlockScale)
    : iterNum(iterNum), blockSize(blockSize), numBlockScale(numBlockScale)
  { opsPerIteration = 0;}

  ~ArithmeticTestBase() {
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

//------------ TEST CLASS FOR BASE POWER MEASUREMENT APPR 1 ---------
template <typename T>
class AddKernel1TestSetSharedMem : public ArithmeticTestBase<T> {
public:

  unsigned int sharedMemRequest;
  float sharedMemScale; 

  AddKernel1TestSetSharedMem(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  AddKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp) {
    ArithmeticTestBase<T>::kernelSetup(deviceProp);

    sharedMemRequest = (unsigned int) (deviceProp.sharedMemPerBlock*sharedMemScale);
  }

  void setSharedMem(float newScale) {
    sharedMemScale = newScale;
  }

  void runKernel() {
      addKernel1_DynamicSharedMem<T><<<this->numBlocks, this->blockSize, sharedMemRequest>>>
                      (this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class MultKernel1TestSetSharedMem : public ArithmeticTestBase<T> {
public:

  unsigned int sharedMemRequest;
  float sharedMemScale; 

  MultKernel1TestSetSharedMem(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  MultKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp) {
    ArithmeticTestBase<T>::kernelSetup(deviceProp);

    sharedMemRequest = (unsigned int) (deviceProp.sharedMemPerBlock*sharedMemScale);
  }

  void setSharedMem(float newScale) {
    sharedMemScale = newScale;
  }

  void runKernel() {
      multKernel_DynamicSharedMem<T><<<this->numBlocks, this->blockSize, sharedMemRequest>>>
                      (this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class FMAKernel1TestSetSharedMem : public ArithmeticTestBase<T> {
public:

  unsigned int sharedMemRequest;
  float sharedMemScale; 

  FMAKernel1TestSetSharedMem(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  FMAKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  //in addition to normal setup, figure out how much shared memory to request
  void kernelSetup(cudaDeviceProp deviceProp) {
    ArithmeticTestBase<T>::kernelSetup(deviceProp);

    sharedMemRequest = (unsigned int) (deviceProp.sharedMemPerBlock*sharedMemScale);
  }

  void setSharedMem(float newScale) {
    sharedMemScale = newScale;
  }

  void runKernel() {
      fmaKernel_DynamicSharedMem<T><<<this->numBlocks, this->blockSize, sharedMemRequest>>>
                      (this->n, this->iterNum, this->d_x);
  }
};

//------------ TEST CASE FOR BASE POWER APPR 2 ---------
template <typename T>
class AddKernel1TestVolatile : public ArithmeticTestBase<T> {
public:
  AddKernel1TestVolatile(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  AddKernel1TestVolatile(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  void runKernel() {
      addKernel1Volatile<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class MultKernel1TestNonVolatile : public ArithmeticTestBase<T> {
public:
  MultKernel1TestNonVolatile(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  MultKernel1TestNonVolatile(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  void runKernel() {
      multKernel1_nonVolitile<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//------------ ADDITION TEST CLASSES ---------
template <typename T>
class AddKernel1Test : public ArithmeticTestBase<T> {
public:
  //this->opsPerIteration = 6;
  AddKernel1Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  AddKernel1Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  void runKernel() {
      addKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class AddKernel2Test : public ArithmeticTestBase<T> {
public:
  AddKernel2Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 9;}
  AddKernel2Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 9;}

  void runKernel() {
      addKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//------------ MULTIPLICATION TEST CLASSES ---------
template <typename T>
class MultKernel1Test : public ArithmeticTestBase<T> {
public:
  MultKernel1Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 6;}
  MultKernel1Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 6;}

  void runKernel() {
      multKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class MultKernel2Test : public ArithmeticTestBase<T> {
public:
  MultKernel2Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 9;}
  MultKernel2Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 9;}

  void runKernel() {
      multKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//------------ FMA TEST CLASSES ---------
template <typename T>
class FmaKernel1Test : public ArithmeticTestBase<T> {
public:
  FmaKernel1Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 9;}
  FmaKernel1Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 9;}

  void runKernel() {
      fmaKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class FmaKernel2Test : public ArithmeticTestBase<T> {
public:
  FmaKernel2Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {this->opsPerIteration = 12;}
  FmaKernel2Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {this->opsPerIteration = 12;}

  void runKernel() {
      fmaKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};








