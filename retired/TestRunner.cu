#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

//run command: 
  //nvcc runtest.cu -L/usr/lib64/nvidia -lnvidia-ml -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp


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


class RunTest {
	/* Constants/enumerations:
	*/
public:
	//int deviceIDNum: GPU device to do all work/sampling on
	int deviceIDNum = 0;

	//desired filename for output
	const char *outputName= "testOutput.txt"; 

	//max number of times to run the test in a row
	int maxTestRuns = 1;

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

  /*
	constructor
		pass kernel functions and output name?
  */
	RunTest() {
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

		//TODO: may need to change stream above or just use gpuStart
		while (cudaEventQuery(kernelFinished) != cudaSuccess) {
		// while (cudaEventQuery(kernelFinished) != cudaSuccess) {
			powerLevel = getPowerUseage();
			tempLevel = getDeviceTemp();
			powerData.push_back(powerLevel);
			tempData.push_back(tempLevel);
		}
		if ((int)powerData.size() == 0) {
			printf("powerdata is empty after sampling\n");
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

		while( (badSampleData || std::abs((int)(curTemp - prevTemp)) >= 1) 
						    && curRun <= maxTestRuns ) 
		{
			printf("  beginning test run %d\n", curRun);
			
		  int n = 1<<18;
		  CUDA_ERROR( cudaMalloc(&d_x, n*sizeof(float)) ); 
		  CUDA_ERROR( cudaMalloc(&d_y, n*sizeof(float)) );
		  createData<<<(n+255)/256, 256>>>(n, d_x, d_y);

			setupSampling();
			runTest();
			runSampling();

			badSampleData = !isDataValid();

			prevTemp = curTemp;
			curTemp = getDeviceTemp();
			curRun++;
		}

		if (maxTestRuns < curRun) {
			printf("maxTestRuns exceeded, sample data may be bad:\n");
			printf("  curTemp - prevTemp = %d\n", (int)curTemp-prevTemp);
			printf("  badSampleData: %d\n", badSampleData);
		}
	}

	/*
	run test on given kernel
	  don't worry about sampling, just test setup/run/cleanup
	  called by getGoodSample
	*/
	void runTest() {
		//TODO: generalize function
	  int n = 1<<18;
	  int iterateNum = 1500000;
		addition6_FP32<<<(n+255)/255, 256>>>(n,iterateNum, d_x, d_y);
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
		//TODO
		return false;
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
		printf("elapsed time: %3.1f\n", elapsedT);
		float timeBetweenSample = elapsedT / powerData.size();
		
		FILE *fp = fopen(outputName, "w+");
		fprintf(fp, "Power(W), Temperature(ºC), Time(ms), elapsed time(ms), number of samples\n");
		fprintf(fp, "%.3lf, %d, %f, %3.1f, %d\n", powerData[0]/1000.0, tempData[0], timeBetweenSample, elapsedT, (int)powerData.size());
		
		for (int i = 1; i < powerData.size(); i++) {
			fprintf(fp, "%.3lf, %d, %f\n", powerData[i]/1000.0, tempData[0], timeBetweenSample*i );
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


}; //end RunTest

int main() {
	printf("calling constructor\n");
	RunTest test = RunTest();
	printf("calling getGoodSample\n");
	test.getGoodSample();
	printf("calling dataToFile\n");
	test.dataToFile();
	return 0;
}

