
//alg 1 and 2 work only for FP32/64 and need register limiting
//checked assembly code with NVCC 9.2
//the only diff b/t alg 1 and 2 is the # of adds
// only verified with following flag:
//   --maxrregcount 15

/*
template <typename T>
__global__
void addAlg1(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  T b, c;
  for (int i = 0; i < iterateNum; i++) {
    b = a + i;
    c = a + b;
    a = c + a;
  }
  x[thread] = a;
}

template <typename T>
__global__
void addAlg2(int n, int iterateNum, T *x) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  T a = x[thread];
  T b = 1000;
  T c = 1000;
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
*/

//algs 3 and 4 work for FP32, FP64, and Int without register limiting

template <typename T>
__global__
void addAlg3(int n, int iterateNum, T *x) {
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
void addAlg4(int n, int iterateNum, T *x) {
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



template <typename T>
__global__
void createData(int n, T *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  T a = 1.0;
  if (i < n) {
    x[i] = a;
  }
}

/*
template <typename T>
class AdditionFP_1 {
public:

  //fields to prepare for kernel call
  T *d_x;
  int n;
  int iterNum;
  int numBlocks;
  int blockSize;


  AdditionFP_1(int blockSize, int iterNum)
    : iterNum(iterNum), blockSize(blockSize) 
  { }

  ~AdditionFP_1() {
    CUDA_ERROR( cudaFree(d_x) );
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    numBlocks = deviceProp.multiProcessorCount * 360;
    n = numBlocks * blockSize;
    CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(T)) ); 
    createData<T><<<numBlocks, blockSize>>>(n, d_x);
  }

  void runKernel() {
        addAlg3<T><<<numBlocks, blockSize>>>(n, iterNum, d_x);
  }

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda error in test class: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};


template <typename T>
class AdditionFP_2 {
public:

  //fields to prepare for kernel call
  T *d_x;
  int n;
  int iterNum;
  int numBlocks;
  int blockSize;


  AdditionFP_2(int blockSize, int iterNum)
    : iterNum(iterNum), blockSize(blockSize) 
  { }

  ~AdditionFP_2() {
    CUDA_ERROR( cudaFree(d_x) );
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    numBlocks = deviceProp.multiProcessorCount * 360;
    n = numBlocks * blockSize;
    CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(T)) ); 
    createData<T><<<numBlocks, blockSize>>>(n, d_x);
  }

  void runKernel() {
        addAlg4<T><<<numBlocks, blockSize>>>(n, iterNum, d_x);
  }

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda error in test class: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};
*/

template <typename T>
class AddBase {
public: 

  T *d_x;
  int n;
  int iterNum;
  int numBlocks;
  int blockSize;

  AddBase(int blockSize, int iterNum)
    : iterNum(iterNum), blockSize(blockSize) 
  { }

  ~AddBase() {
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

template <typename T>
class AddAlg3Test : public AddBase<T> {
public:
  //using AddBase<T>::AddBase;
  AddAlg3Test(int blockSize, int iterNum) : AddBase<T>(blockSize, iterNum) 
  {}
  void runKernel() {
      addAlg3<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};

template <typename T>
class AddAlg4Test : public AddBase<T> {
public:
  //using AddBase<T>::AddBase(int, int);
  AddAlg4Test(int blockSize, int iterNum) : AddBase<T>(blockSize, iterNum) 
  {}

  void runKernel() {
      addAlg4<T><<<this->numBlocks, this->blockSize>>>(this->n, this->iterNum, this->d_x);
  }
};











