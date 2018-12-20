#include "testFramework.cu"
#include "arithmeticTests.h"
#include "memoryTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include <tuple>
#include <vector>

template <typename kernel>
void runTestGeneric(int, int, const char*);
template <typename kernel>
void runTestGeneric_setBlockScale(int, int, const char*, int);

template <typename kernel>
void runL1Test(int, int, const char*, const char*);
template <typename kernel>
void runL2Test(int, int, const char*, const char*);

template <typename T>
void runGlobalTest(int, int, const char*, const char*);

template <typename kernel>
void runSharedMemTest(int, int, const char*, const char*);


int main() {
//  std::string storagePath = setupStoragePath(argc, argv);

  int blockSize = 256;
  int iterationsSmall = 200000;
  int iterationsBig = 200000000;
  
  // std::string out1;
  // std::string out2;
  // out1 = storagePath + std::string("outputAddFP32_1.csv");
  // out2 = storagePath + std::string("outputAddFP32_2.csv");
 
  runTestGeneric_setBlockScale<L1MemTest1<float>>(iterationsSmall*3, blockSize, "data/outputL1ReadTest_1.csv", 100);
  runTestGeneric_setBlockScale<L1MemTest2<float>>(iterationsSmall*3, blockSize, "data/outputL1ReadTest_2.csv", 100);

  runTestGeneric_setBlockScale<L2MemReadTest1<float>>(iterationsSmall*3, blockSize, "data/outputL2ReadTest_1.csv", 100);
  runTestGeneric_setBlockScale<L2MemReadTest2<float>>(iterationsSmall*3, blockSize, "data/outputL2ReadTest_2.csv", 100);

  runTestGeneric_setBlockScale<SharedMemReadTest1<float>>(iterationsSmall*4, blockSize, "data/outputSharedReadTest_1.csv", 100);
  runTestGeneric_setBlockScale<SharedMemReadTest2<float>>(iterationsSmall*4, blockSize, "data/outputSharedReadTest_2.csv", 100);


  runGlobalTest<float>(1000, blockSize, "data/outputGlobalReadTest_1.csv", "data/outputGlobalReadTest_2.csv");

  // printf("---- beginning L1 Testing ----\n"); 
  // runL1Test<L1MemTest1<T>>(iterations, blockSize, "tmp1.csv");
  // printf("---- test end ----\n");
  
  // printf("---- beginning L2 Testing ----\n"); 
  // runL2Test<L2MemReadTest1<float>>(iterations, blockSize, "tmp1.csv");
  // printf("---- test end ----\n");
  
  // printf("---- beginning Global Memory Testing ----\n"); 
  // runGlobalTest<GlobalMemTest1<float>>(iterations, blockSize, "tmp1.csv");
  // printf("---- test end ----\n");

  // printf("---- beginning Shared Memory Testing ----\n"); 
  // runSharedMemTest<SharedMemReadTest1<float>>(iterations, blockSize, "tmp1.csv");
  // printf("---- test end ----\n");

  return 0;
}


template <typename kernel>
void runTestGeneric(int iterNum, int blockSize, const char* outputName) 
{
  printf("Starting Kernel: '%s'\n", outputName);
  kernel test1(blockSize, iterNum);
  TestRunner<kernel> tester1(&test1, outputName, 0.1);
  tester1.getGoodSample();
  tester1.dataToFile();
  printf("Kernel '%s' finished\n", outputName);
}

template <typename kernel>
void runTestGeneric_setBlockScale(int iterNum, int blockSize, const char* outputName, int blockScale)
{
  printf("Starting Kernel: '%s'\n", outputName);
  kernel test1(blockSize, iterNum, blockScale);
  TestRunner<kernel> tester1(&test1, outputName, 0.1);
  tester1.getGoodSample();
  tester1.dataToFile();
  printf("Kernel '%s' finished\n", outputName);
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



template <typename T>
void runGlobalTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Global Kernel1\n");
  GlobalMemTest1<T> test1(blockSize, iterNum);
  TestRunner<GlobalMemTest1<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  tester1.dataToFile();
  printf("Kernel 1 finished\n");

  printf("Starting Global Kernel2\n");
  GlobalMemTest2<T> test2(blockSize, iterNum);
  TestRunner<GlobalMemTest2<T>> tester2(&test2, outputName2);
  tester2.getGoodSample();
  tester2.dataToFile();
  printf("Kernel 2 finished\n");
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
