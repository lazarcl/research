/*********************************************
******************Pair #1*********************
**********************************************/

  /* Ideas:
  keep input small-ish
  loop
  don't read much data
  difference should only be additions, nothing else
  */

__global__
void addFP32alg1(int n, int iterateNum, float *x, float *y) {

  int start = blockIdx.x*blockDim.x + threadIdx.x;

  int l = 0;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    l = (l+1) % n;
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];
  }

}

__global__ 
void addFP32alg2(int n, int iterateNum, float *x, float *y) {
  int start = blockIdx.x*blockDim.x + threadIdx.x;
  int l = 0;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    l = (l+1) % n;
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];


    //6 more additions?:
    //no extra lookups
    //l-1, l-2 already calculated, so no subtraction needed
    //no new stores after optimization:
    //  x[l] store will only happen once, not 3 times. 
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];
  }

}

/*********************************************
******************Pair #2*********************
**********************************************/


__global__
void addFP32alg3(int n, int iterateNum, float *x, float *y) {

  int start = blockIdx.x*blockDim.x + threadIdx.x;

  int l = 0;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    l = (l+1) % n;
    x[l] = y[l];
  }

}

__global__
void addFP32alg4(int n, int iterateNum, float *x, float *y) {
  int start = blockIdx.x*blockDim.x + threadIdx.x;
  int l = 0;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    l = (l+1) % n;
    x[l] = y[l] + y[l] + y[l];
  }

}

/*********************************************
******************Pair #3*********************
**********************************************/


__global__
void addFP32alg5(int n, int iterateNum, float *x, float *y) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  volatile float a = x[thread];
  volatile float b = 1000;
  volatile float c = 1000;
  for (int i = 0; i < iterateNum; i++) {
    b = a + i;
    c = a + b;
    a = c + a;

  }
  x[thread] = a;
}


__global__
void addFP32alg6(int n, int iterateNum, float *x, float *y) {
  int thread = blockIdx.x*blockDim.x + threadIdx.x;
  volatile float a = x[thread];
  volatile float b = 1000;
  volatile float c = 1000;
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


__global__
void createData(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  volatile float a = 1.0;
  volatile float b = 2.0;
  if (i < n) {
    x[i] = a;
    y[i] = b;
  }
}

class AdditionFP32 {
public:

  //fields to prepare for kernel call
  float *d_x, *d_y;
  int n;
  int iterNum;
  int numBlocks;
  int blockSize;


  AdditionFP32() {
    blockSize = 256;
    iterNum = 1000000;
  }

//destructor
  ~AdditionFP32() {
    CUDA_ERROR( cudaFree(&d_x) );
    CUDA_ERROR( cudaFree(&d_y) );
  }

  void kernelSetup(cudaDeviceProp deviceProp) {
    numBlocks = deviceProp.multiProcessorCount * 360;
    n = numBlocks * blockSize;
    CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(float)) ); 
    CUDA_ERROR( cudaMalloc(&d_y, n*sizeof(float)) );
    createData<<<numBlocks, blockSize>>>(n, d_x, d_y);
  }

  void runKernel() {
        addFP32alg6<<<numBlocks, blockSize>>>(n, iterNum, d_x, d_y);
  }

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda Error: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};
















