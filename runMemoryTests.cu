#include "testFramework.cu"
#include "arithmeticTests.cu"
#include "memoryTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include <tuple>
#include <vector>

template <typename T>
void runGlobalTest(int, int, const char*, const char*);


void main() {
  std::string storagePath = setupStoragePath(argc, argv);

  int blockSize = 256;
  int iterations = 10;
  
  std::string out1;
  std::string out2;
  printf("---- beginning FP32 Add Testing ----\n"); 
  // out1 = storagePath + std::string("outputAddFP32_1.csv");
  // out2 = storagePath + std::string("outputAddFP32_2.csv");
  runGlobalTest<float>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  printf("---- test end ----\n");

}


template <typename T>
void runGlobalTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  GlobalMemTest<T> test1(blockSize, iterNum);
  TestRunner<GlobalMemTest<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();

  printf("Kernel 1 finished\n");

  // printf("Starting Kernel2\n");
  // AddKernel2Test<T> test2(blockSize, iterNum);
  // TestRunner<AddKernel2Test<T>> tester2(&test2, outputName2);
  // tester2.getGoodSample();
  // tester2.dataToFile();

  // printf("Kernel 2 finished\n");
}
