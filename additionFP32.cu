// #include <stdio.h>
// #include <nvml.h>
#include "nvmlPower.hpp"
#include "nvmlPower.cpp"


//run command: 
  //nvcc additionFP32.cu -L/usr/lib64/nvidia -lnvidia-ml -lpthread -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp


__global__
void addition1_FP32(int n, int iterateNum, float *x, float *y) {

  int start = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    x[i] = x[i] + y[i] + y[i-1] + y[i-2];
  }

}

__global__
void addition2_FP32(int n, int iterateNum, float *x, float *y) {
  /* Ideas:
  keep input small-ish
  loop
  don't read much data
  difference should only be additions, nothing else
  */
  int start = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = start; i < iterateNum /*&& i < n && i >= 0*/; i++) {
    x[i] = x[i] + y[i] + y[i-1] + y[i-2];


    //6 more additions?:
    //no extra lookups
    //i-1, 1-2 already calculated, so no subtraction needed
    //no new stores after optimization:
    //  x[i] store will only happen once, not 3 times. 
    x[i] = x[i] + y[i] + y[i-1] + y[i-2];
    x[i] = x[i] + y[i] + y[i-1] + y[i-2];
  }

}


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
    printf("Attempt writing time Elapsed in '%s' failed. Error: ", outputName);
    perror("");
    printf("Terminating...");
    exit(0);
  }
  fseek(fp, 25, SEEK_SET);
  fprintf(fp, "timeElapsed:, %3.1f \n", time);
  fclose(fp);
  printf("Algorithm finished, results saved in %s\n", outputName);
}



int main(int argc, char* argv[]) 
{

  if (argc !=3) {
    printf("expected 2 int or float arguments. Quitting.\n");
    exit(0);
  }

  int N = 1<<15;
  int iterateNum = 100000;

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  HANDLE_ERROR( cudaMalloc(&d_x, N*sizeof(float)) ); 
  HANDLE_ERROR( cudaMalloc(&d_y, N*sizeof(float)) );

  createData<<<(N+255)/255, 256>>>(N, d_x, d_y, atof(argv[1]), atof(argv[2]));


  runAnalysis("Power_data_add32_Alg1.txt", addition1_FP32, N, iterateNum, d_x, d_y);
  HANDLE_ERROR( cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost) );
  
  runAnalysis("Power_data_add32_Alg2.txt", addition2_FP32, N, iterateNum, d_x, d_y);
  HANDLE_ERROR( cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost) );


  HANDLE_ERROR( cudaFree(d_x) );
  HANDLE_ERROR( cudaFree(d_y) );
  free(x);
  free(y);
}





