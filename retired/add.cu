// #include <stdio.h>
// #include <nvml.h>
#include "nvmlPower.hpp"
#include "nvmlPower.cpp"


//run command: 
  //nvcc add.cu -L/usr/lib64/nvidia -lnvidia-ml -lpthread -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp



// nvmlReturn_t HANDLE_ERROR_NVML(nvmlReturn_t e) {
//   if (e != NVML_SUCCESS) {
//     printf("nvml error: \"%s\" \n", nvmlErrorString(e));
//     exit(EXIT_FAILURE);
//   }
//   return e;
// }

// nvmlDevice_t startNvml() {
//   HANDLE_ERROR_NVML( nvmlInit() );
//   nvmlDevice_t deviceID;
//   HANDLE_ERROR_NVML( nvmlDeviceGetHandleByIndex(0, &deviceID) );
//   return deviceID;
// }

// void endNvml() {
//   nvmlReturn_t e = nvmlShutdown();
//   if ( e != NVML_SUCCESS ) {
//     printf("nvml shutdown failure: \"%s\"\n", nvmlErrorString(e));
//   }
// }

// void measurePower(nvmlDevice_t deviceID) {
//   unsigned int power = 0;
//   HANDLE_ERROR_NVML( nvmlDeviceGetPowerUsage(deviceID, &power) );
//   printf("measured power was: \"%d\"\n", power);
// }

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__global__
void saxpyLoop(int n, float a, float *x, float *y)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  // if (i=1) nvmlPSUInfo_t::power
  for (int i = idx; i < idx+10000 && i < n; i++) {
    y[i] = a*x[i] + y[i];
  }
}


__global__
void createData(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

void HANDLE_ERROR(cudaError_t e) {
  if (e != cudaSuccess) {
    printf("cuda Error: \"%s\"\n", cudaGetErrorString(e));
  }
} 

int main(void) 
{
  int N = 1<<29;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  HANDLE_ERROR( cudaMalloc(&d_x, N*sizeof(float)) ); 
  HANDLE_ERROR( cudaMalloc(&d_y, N*sizeof(float)) );
  createData<<<(N+255)/255, 256>>>(N, d_x, d_y);


  float time;
  cudaEvent_t start, stop;
  HANDLE_ERROR( cudaEventCreate(&start) );
  HANDLE_ERROR( cudaEventCreate(&stop) );
  
  nvmlAPIRun();

  HANDLE_ERROR( cudaEventRecord(start) );
  saxpyLoop<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  HANDLE_ERROR( cudaEventRecord(stop, 0) );

  HANDLE_ERROR( cudaEventSynchronize(stop) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
  
  nvmlAPIEnd();

  HANDLE_ERROR( cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost) );

  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++)
  //   maxError = max(maxError, abs(y[i]-4.0f));

  printf("Elapsed time: %3.1f ms\n", time); 
  // printf("Max error:    %.2f\n", maxError);

  HANDLE_ERROR( cudaFree(d_x) );
  HANDLE_ERROR( cudaFree(d_y) );
  free(x);
  free(y);
}





