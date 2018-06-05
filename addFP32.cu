//#include <testFramework.cu>
//#include <baseTestClass.cu>



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


class AdditionFP32 : BaseTestClass {


  //references to the different addition run algs. 5/6 is best
  K alg1 = &addFP32alg1;
  K alg2 = &addFP32alg2;
  K alg3 = &addFP32alg3;
  K alg4 = &addFP32alg4;
  K alg5 = &addFP32alg5;
  K alg6 = &addFP32alg6;

  //the device id to run tests on
  // int deviceID;

  //framework to run and sample the kernel
  // TestRunner tester;

  //fields to prepare for kernel call
  float *d_x, *d_y;
  int n = 1<<18;
  int iterNum = 1000000;
  int numBlocks = (n+255)/256;
  int blockSize = 256;


  AdditionFP32() {
    // tester = TestRunner("testOutput.txt");
  }

  void kernelSetup() {
    CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(float)) ); 
    CUDA_ERROR( cudaMalloc(&d_y, n*sizeof(float)) );
    createData<<<(n+255)/256, 256>>>(n, d_x, d_y);
  }

  void runKernel(int num) {
    switch(num) {
      case 6 : 
        alg6<<<numBlocks, blockSize>>>(n, iterNum, d_x, d_y);
        break;
    }
  }

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda Error: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

};
















