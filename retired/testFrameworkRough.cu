// #include <stdio.h>
// #include <nvml.h>
#include "nvmlPower.hpp"
#include "nvmlPower.cpp"


//run command: 
  //nvcc additionFP32.cu -L/usr/lib64/nvidia -lnvidia-ml -lpthread -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp


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
void addition1_FP32(int n, int iterateNum, float *x, float *y) {

  int start = blockIdx.x*blockDim.x + threadIdx.x;

  int l = 0;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    l = (l+1) % n;
    x[l] = x[l] + y[l] + y[l-1] + y[l-2];
  }

}

__global__
void addition2_FP32(int n, int iterateNum, float *x, float *y) {
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


/*
Loop:
  x[i] = y[i]
---vs.---
loop2:
  x[i] = + y[i] + y[i] + y[i]

*/
__global__
void addition3_FP32(int n, int iterateNum, float *x, float *y) {

  int start = blockIdx.x*blockDim.x + threadIdx.x;

  int l = 0;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    l = (l+1) % n;
    x[l] = y[l];
  }

}

__global__
void addition4_FP32(int n, int iterateNum, float *x, float *y) {
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
void addition5_FP32(int n, int iterateNum, float *x, float *y) {
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
void addition6_FP32(int n, int iterateNum, float *x, float *y) {
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

/*********************************************
**********************************************
**********************************************/


__global__
void createData(int n, float *x, float *y, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = a;
    y[i] = b;
  }
}


void HANDLE_ERROR(cudaError_t e) {
  if (e != cudaSuccess) {
    printf("cuda Error: \"%s\"\n", cudaGetErrorString(e));
  }
} 


void runAnalysis(const char *outputName, 
          void gpuFunc(int, int, float *, float *), 
          int n, int iterateNum, float *d_x, float *d_y) {

  float time;
  cudaEvent_t start, stop;
  HANDLE_ERROR( cudaEventCreate(&start) );
  HANDLE_ERROR( cudaEventCreate(&stop) );
  
  nvmlAPIRun(outputName);

  HANDLE_ERROR( cudaEventRecord(start) );
  gpuFunc<<<(n+255)/256, 256>>>(n, iterateNum, d_x, d_y);
  HANDLE_ERROR( cudaEventRecord(stop, 0) );

  HANDLE_ERROR( cudaEventSynchronize(stop) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
  
  nvmlAPIEnd();

  FILE *fp = fopen(outputName, "r+");
  if (fp == NULL) {
    printf("Attempt at writing 'time Elapsed' in '%s' failed. Error: ", outputName);
    perror("");
    printf("Terminating...");
    exit(0);
  }
  fseek(fp, 30, SEEK_SET);
  fprintf(fp, "timeElapsed:, %3.1f\n", time);
  fclose(fp);
  printf("Algorithm finished, results saved in %s\n", outputName);
}



int main(int argc, char* argv[]) 
{

  if (argc !=3) {
    printf("expected 2 int or float arguments. Quitting.\n");
    exit(0);
  }

  int N = 1<<18;
  int iterateNum = 10000000;



  // int deviceId = 0;
  // cudaError_t cudaRet;
  // cudaDeviceProp deviceProp;
  // cudaRet = cudaGetDeviceProperties ( &deviceProp, deviceId );
  // if (cudaRet != cudaSuccess) {
  //   printf("get deviceProp failed: %s\n", cudaGetErrorString(cudaRet));
  // }
  // int numBlocks = 360 * deviceProp.multiProcessorCount;


  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  HANDLE_ERROR( cudaMalloc(&d_x, N*sizeof(float)) ); 
  HANDLE_ERROR( cudaMalloc(&d_y, N*sizeof(float)) );

  createData<<<(N+255)/256, 256>>>(N, d_x, d_y, atof(argv[1]), atof(argv[2]));


  runAnalysis("Power_data_add32_Alg5.txt", addition5_FP32, N, iterateNum, d_x, d_y);
  HANDLE_ERROR( cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost) );
  
  runAnalysis("Power_data_add32_Alg6.txt", addition6_FP32, N, iterateNum, d_x, d_y);
  HANDLE_ERROR( cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost) );


  HANDLE_ERROR( cudaFree(d_x) );
  HANDLE_ERROR( cudaFree(d_y) );
  free(x);
  free(y);
}





