
//mult, add and FMA Kernels work for FP32, FP64, and Int 
//  without register limiting


//------------ EXPERIMENT VOLATILE KERNEL FOR BASEPOWER 2 ---------
template <typename T>
__global__
void addKernel1Volatile(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  volatile T b = 2;
  volatile T c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + i;
    c = a + b;
    a = c + a;
    c = b + a;
    b = c + a;
    a = b + c;
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
  T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + i;
    c = a + b;
    a = c + a;
    c = b + a;
    b = c + a;
    a = b + c;
  }
  x[thread] = a;
}

//------------ ADDITION KERNELS ---------
template <typename T>
__global__
void addKernel1(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + i;
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
  T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a + i;
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
  T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a * i;
    c = a * b;
    a = c * a;
    c = b * a;
    b = c * a;
    a = b * c;
  }
  x[thread] = a;
}


template <typename T>
__global__
void multKernel2(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  T b = 2, c = 2;
  for (int i = 0; i < iterateNum; i++) {
    b = a * i;
    c = a * b;
    a = c * a;
    c = b * a;
    b = c * a;
    a = b * c;
    c = b * a;
    b = c * a;
    a = b * c;
  }
  x[thread] = a;
}


//------------ FMA KERNELS ---------
template <typename T>
__global__
void fmaKernel1(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  T b, c;
  for (int i = 0; i < iterateNum; i++) {
    b = a*i + i;
    c = a*b + a;
    a = c*b + a;
    c = b*a + c;
    b = a*b + b;
    a = b*c + a;
    c = b*a + c;
    b = a*b + b;
    a = b*c + a;
  }
  x[thread] = a;
}

template <typename T>
__global__
void fmaKernel2(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  T b, c;
  for (int i = 0; i < iterateNum; i++) {
    b = a*i + i;
    c = a*b + a;
    a = c*b + a;
    c = b*a + c;
    b = a*b + b;
    a = b*c + a;
    c = b*a + c;
    b = a*b + b;
    a = b*c + a;
    
    c = b*a + c;
    b = a*b + b;
    a = b*c + a;    
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
  int getTotalOperationCount() {
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
  {}
  AddKernel1TestSetSharedMem(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

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

//------------ TEST CASE FOR BASE POWER APPR 2 ---------
template <typename T>
class AddKernel1TestVolatile : public ArithmeticTestBase<T> {
public:
  AddKernel1TestVolatile(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  AddKernel1TestVolatile(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

  void runKernel() {
      addKernel1Volatile<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//------------ ADDITION TEST CLASSES ---------
template <typename T>
class AddKernel1Test : public ArithmeticTestBase<T> {
public:
  opsPerIteration = 6;
  AddKernel1Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  AddKernel1Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

  void runKernel() {
      addKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class AddKernel2Test : public ArithmeticTestBase<T> {
public:
  AddKernel2Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  AddKernel2Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

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
  {}
  MultKernel1Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

  void runKernel() {
      multKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class MultKernel2Test : public ArithmeticTestBase<T> {
public:
  MultKernel2Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  MultKernel2Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

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
  {}
  FmaKernel1Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

  void runKernel() {
      fmaKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class FmaKernel2Test : public ArithmeticTestBase<T> {
public:
  FmaKernel2Test(int blockSize, int iterNum) 
      : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  FmaKernel2Test(int blockSize, int iterNum, int numBlockScale) 
      : ArithmeticTestBase<T>(blockSize, iterNum, numBlockScale) 
  {}

  void runKernel() {
      fmaKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};








