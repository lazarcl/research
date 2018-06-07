
//mult, add and FMA Kernels work for FP32, FP64, and Int 
//  without register limiting


//------------ MULTIPLICATION KERNELS ---------
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

  ArithmeticTestBase(int blockSize, int iterNum)
    : iterNum(iterNum), blockSize(blockSize) 
  { }

  ~ArithmeticTestBase() {
    CUDA_ERROR( cudaFree(d_x) );
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    numBlocks = deviceProp.multiProcessorCount * 360;
    n = numBlocks * blockSize;
    CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(T)) ); 
    createData<T><<<numBlocks, blockSize>>>(n, d_x);
  }

  void runKernel();

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda error in test class: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};


//------------ ADDITION TEST CLASSES ---------
template <typename T>
class AddKernel1Test : public ArithmeticTestBase<T> {
public:
  AddKernel1Test(int blockSize, int iterNum) : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  void runKernel() {
      addKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class AddKernel2Test : public ArithmeticTestBase<T> {
public:
  AddKernel2Test(int blockSize, int iterNum) : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}

  void runKernel() {
      addKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//------------ MULTIPLICATION TEST CLASSES ---------
template <typename T>
class MultKernel1Test : public ArithmeticTestBase<T> {
public:
  MultKernel1Test(int blockSize, int iterNum) : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  void runKernel() {
      multKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class MultKernel2Test : public ArithmeticTestBase<T> {
public:
  MultKernel2Test(int blockSize, int iterNum) : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}

  void runKernel() {
      multKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};


//------------ FMA TEST CLASSES ---------
template <typename T>
class FmaKernel1Test : public ArithmeticTestBase<T> {
public:
  FmaKernel1Test(int blockSize, int iterNum) : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}
  void runKernel() {
      fmaKernel1<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class FmaKernel2Test : public ArithmeticTestBase<T> {
public:
  FmaKernel2Test(int blockSize, int iterNum) : ArithmeticTestBase<T>(blockSize, iterNum) 
  {}

  void runKernel() {
      fmaKernel2<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};








