

template <typename T>
__global__
void addFPalg5(int n, int iterateNum, T *x) {
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
void addFPalg6(int n, int iterateNum, T *x) {
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

template <typename T>
__global__
void createData(int n, T *x, T *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  T a = 1.0;
  if (i < n) {
    x[i] = a;
  }
}

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
        addFPalg5<T><<<numBlocks, blockSize>>>(n, iterNum, d_x);
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
        addFPalg6<T><<<numBlocks, blockSize>>>(n, iterNum, d_x);
  }

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda error in test class: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};
















