#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include "templateAddFP.cu"

//run command: 
  //nvcc testFramework.cu -L/usr/lib64/nvidia -lnvidia-ml -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp

template <class K>
class TestRunner {
public:

  //int deviceIDNum: GPU device to do all work/sampling on
  const int deviceIDNum = 0;

  //max number of times to attempt getting a good sample
  const int maxTestRuns = 15;

  //number of samples to igore from beg and end while analyzing data
  const int ignoreSampleCount = 50;

  //device properties
  cudaDeviceProp deviceProp;

  //desired filename for output
  const char *outputName; 

  //hold data from samples in vectors
  std::vector<float> powerData;
  std::vector<int> tempData;

  //the deviceID in nvml format
  nvmlDevice_t nvmlDeviceID;

  //the result code of the previous nvml call
  nvmlReturn_t nvmlResult;

  //CUDA timing events to clalculate test kernel's runtime
  cudaEvent_t gpuStart, gpuStop, kernelFinished;

  //TODO: remove later. for development testing
  float *d_x, *d_y;

  //class that holds the kernel to run
  K *testClass;

  TestRunner(K *t, const char *outputName) : outputName(outputName) {
    testClass = t;
    
    CUDA_ERROR( cudaSetDevice(deviceIDNum) );
    CUDA_ERROR( cudaGetDeviceProperties(&deviceProp, deviceIDNum) );
    
    nvmlResult = nvmlInit();
    if ( nvmlResult != NVML_SUCCESS )
    {
      printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
      exit(0);
    }
    nvmlResult = nvmlDeviceGetHandleByIndex(deviceIDNum, &nvmlDeviceID);
    if (nvmlResult != NVML_SUCCESS) {
      printf("failed getting device handle by index: %s\n", nvmlErrorString(nvmlResult));
      exit(0);
    }
  }

  /*
  start measurement of device's power/temp
    initial setup of nvml and vars
  */
  void setupSampling()
  {

    //reserve about 6 min of sampling space
    powerData.clear();
    powerData.reserve(750000);  
    tempData.clear();
    tempData.reserve(750000);  

    CUDA_ERROR( cudaEventCreate(&gpuStart) );
    CUDA_ERROR( cudaEventCreate(&gpuStop) );
    CUDA_ERROR( cudaEventRecord(gpuStart) );
  }


  /*
  takeMeasurements
    loop that samples until kernel has finished
    return bool - true: good sample graph, false: bad samples
  */
  void runSampling() {
    unsigned int powerLevel, tempLevel;
    CUDA_ERROR( cudaEventCreate(&kernelFinished) );
    CUDA_ERROR( cudaEventRecord(kernelFinished) ); 

    while (cudaEventQuery(kernelFinished) != cudaSuccess) {
      powerLevel = getPowerUseage();
      tempLevel = getDeviceTemp();
      powerData.push_back(powerLevel);
      tempData.push_back(tempLevel);
    }
    CUDA_ERROR( cudaEventRecord(gpuStop, 0) );
    CUDA_ERROR( cudaEventSynchronize(gpuStop) );
  }


  /*
  get a good test reading
    keep running test until:
      good reading && temp was constant
      haven't looped too many times
    warn if test case is not long enough
    print each time test case is run
    if loop runs to maxItr, then print to file anyways
  */
  
  void getGoodSample() {
    unsigned int prevTemp = 0;
    unsigned int curTemp = 0;
    bool badSampleData = true;
    int curRun = 1;

    testClass->kernelSetup(deviceProp);

    while( badSampleData || std::abs((int)(curTemp - prevTemp)) >= 1) { 
      if (curRun > maxTestRuns ) {
        break;
      }
      printf("  beginning test run %d\n", curRun);
      //take temp measurement again here?
      setupSampling();
      testClass->runKernel();
      runSampling();

      prevTemp = curTemp;
      curTemp = getDeviceTemp();

      if (std::abs((int)(curTemp-prevTemp)) >= 1) {
        printf("    temp still changing\n");
      } else {
        badSampleData = !isDataValid();
      }
      curRun++;
    }

    if (maxTestRuns < curRun) {
      printf("maxTestRuns exceeded, sample data may be bad. Debug info:\n");
      printf("  curTemp - prevTemp = %d\n", (int)curTemp-prevTemp);
      printf("  badSampleData: %d\n", badSampleData);
    }
  }

  /*
  analyze test's sampling
    make sure graph is flat and level
    1%-3% tolerance in graph's height
      maybe allow small downward spikes 5ms in width
      very beg and end can ramp up/down though
  return true(good enough) / false(bad samples)
  */
  bool isDataValid() {
    if ((int)powerData.size() < 2*ignoreSampleCount) {
      printf("Only %d samples in previous run\n", (int)powerData.size());
      return false;
    }
    int startPt = ignoreSampleCount;
    int endPt = (int) powerData.size() - ignoreSampleCount;
    double avg = getPowerAvg(startPt, endPt);

    int badStreak = 0;
    double curSample;
    for (int i = startPt; i < endPt; i++) {
      curSample = (double) powerData[i];
      if (std::abs((curSample-avg)/avg) > 0.03) {

        if (badStreak++ > 250) {
          printf("    power samples not consistant enough. Made it %d%% through data\n", 100*i/endPt);
          return false;
        }

      } else {
        badStreak = 0;
      }
    }

    return true;
  }

  /*
  find the avg power value by random sampling
  */
  double getPowerAvg(int startPt, int endPt) {
    double total = 0;
    for (int i = startPt; i < endPt; i++) {
      total += (double)powerData[i];
    }
    return total / (endPt - startPt);
  }

  /*
  write data to file
  file is .txt but organized as .csv
    first row: "Power(W), Temperature(ºC), Time(ms), elapsed time(ms), number of samples"
    second row: "xxx, yyy, zzz, aaa, bb"
    third row->end: "xxxx, yyyy, zzzz"
  */
  void dataToFile() {
    float elapsedT;
    CUDA_ERROR( cudaEventElapsedTime(&elapsedT, gpuStart, gpuStop));
    float timeBetweenSample = elapsedT / powerData.size();
    
    FILE *fp = fopen(outputName, "w+");
    if (fp == NULL) {
      printf("Attempt at opening '%s' failed. Error: ", outputName);
      perror("");
      printf("Terminating...");
      exit(0);
    }
    fprintf(fp, "Power(W), Temperature(ºC), Time(ms), elapsed time(ms), number of samples\n");
    fprintf(fp, "%.3lf, %d, %f, %3.1f, %d\n", powerData[0]/1000.0, tempData[0], timeBetweenSample, elapsedT, (int)powerData.size());
    
    for (int i = 1; i < powerData.size(); i++) {
      fprintf(fp, "%.3lf, %d, %f\n", powerData[i]/1000.0, tempData[0], timeBetweenSample*(i+1) );
    }
    fclose(fp);
  }

  /*
  find number of devices
  */
  void numOfDevices() {
    unsigned int deviceCount;
    nvmlResult = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != nvmlResult)
    {
      printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
      exit(0);
    }
    printf("There are %d GPUs connected, ensure the correct one is selected\n", deviceCount);
  }

  /*
  get device temp
    return temp as unsigned int
    return 0 if there was a sampling error
  */
  unsigned int getDeviceTemp() {
    unsigned int tempLevel;
    nvmlResult = nvmlDeviceGetTemperature(nvmlDeviceID, NVML_TEMPERATURE_GPU, &tempLevel);
    if (nvmlResult != NVML_SUCCESS) {
      tempLevel = 0;
      printf("NVML temp measurement fail: %s\n", nvmlErrorString(nvmlResult));
    }
    return tempLevel;
  }

  unsigned int getPowerUseage() {
    unsigned int powerLevel;
    nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
    if (nvmlResult != NVML_SUCCESS) {
      printf("NVML power measurement fail: %s\n", nvmlErrorString(nvmlResult));
    }
    return powerLevel;
  }

  void CUDA_ERROR(cudaError_t e) {
    if (e != cudaSuccess) {
      printf("cuda Error: \"%s\"\n", cudaGetErrorString(e));
    }
  } 

}; //end TestRunner

template <typename T>
void runAddFPTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("creating AdditionFP32_1 TestClass\n");
  AdditionFP_1<T> test1(blockSize, iterNum);
  printf("creating TestRunner obj\n");
  TestRunner<AdditionFP_1<T>> tester1(&test1, outputName1);
  printf("calling getGoodSample\n");
  tester1.getGoodSample();
  printf("calling dataToFile\n");
  tester1.dataToFile();

  printf("AdditionFP32_1 finished\n");

  printf("ceating AdditionFP32_2 TestClass\n");
  AdditionFP_2<T> test2(blockSize, iterNum);
  printf("creating TestRunner obj\n");
  TestRunner<AdditionFP_2<T>> tester2(&test2, outputName2);
  printf("calling getGoodSample\n");
  tester2.getGoodSample();
  printf("calling dataToFile\n");
  tester2.dataToFile();

  printf("AdditionFP32_2 finished\n");
}

int main() {
  printf("---- beginning FP32 Add Testing ----\n"); 
  runAddFPTest<float>(6000000, 256, "data/outputAddFP32_1.txt", "data/outputAddFP32_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning FP64 Add Testing ----\n");
  runAddFPTest<double>(6000000, 256, "data/outputAddFP64_1.txt", "data/outputAddFP64_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning Int32 Add Testing ---\n");
  runAddFPTest<int>(6000000, 256, "data/outputAddInt32_1.txt", "data/outputAddInt32_2.txt");
  printf("---- test end ----\n");
  return 0;
}
