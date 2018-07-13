#include "testFramework.cu"
#include "arithmeticTests.h"
#include "memoryTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include <tuple>
#include <vector>

template <typename kernel>
void runTestGeneric(int, int, const char*, const char*);
template <typename kernel>
void runL1Test(int, int, const char*, const char*);
template <typename kernel>
void runL2Test(int, int, const char*, const char*);
template <typename kernel>
void runGlobalTest(int, int, const char*, const char*);
template <typename kernel>
void runSharedMemTest(int, int, const char*, const char*);


int main() {
//  std::string storagePath = setupStoragePath(argc, argv);

  int blockSize = 256;
  int iterations = 1;
  
  std::string out1;
  std::string out2;
  // out1 = storagePath + std::string("outputAddFP32_1.csv");
  // out2 = storagePath + std::string("outputAddFP32_2.csv");
  
  runTestGeneric<L2MemTest1<float>>(iterations, blockSize, "tmp1.csv", "tmp2.csv");

  // printf("---- beginning L1 Testing ----\n"); 
  // runL1Test<L1MemTest1<T>>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");
  
  // printf("---- beginning L2 Testing ----\n"); 
  // runL2Test<L2MemTest1<float>>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");
  
  // printf("---- beginning Global Memory Testing ----\n"); 
  // runGlobalTest<GlobalMemTest1<float>>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");

  // printf("---- beginning Shared Memory Testing ----\n"); 
  // runSharedMemTest<SharedMemTest1<float>>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");

  return 0;
}


template <typename kernel>
void runTestGeneric(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  kernel test1(blockSize, iterNum);
  TestRunner<kernel> tester1(&test1, outputName1);
  tester1.getGoodSample();
  tester1.dataToFile();
  printf("Kernel 1 finished\n");
}



template <typename kernel>
void runL1Test(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  L1MemTest1<T> test1(blockSize, iterNum);
  TestRunner<L1MemTest1<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
}

template <typename kernel>
void runL2Test(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  L2MemTest1<T> test1(blockSize, iterNum);
  TestRunner<L2MemTest1<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
}



template <typename kernel>
void runGlobalTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  GlobalMemTest1<T> test1(blockSize, iterNum);
  TestRunner<GlobalMemTest1<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
}


template <typename kernel>
void runSharedMemTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  SharedMemTest1<T> test1(blockSize, iterNum);
  TestRunner<SharedMemTest1<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
}
