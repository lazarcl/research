#include "testFramework.cu"
#include "arithmeticTests.h"
#include "memoryTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include <tuple>
#include <vector>

template <typename kernel, typename T>
void runTestGeneric(int, int, const char*, const char*);
template <typename T>
void runL1Test(int, int, const char*, const char*);
template <typename T>
void runL2Test(int, int, const char*, const char*);
template <typename T>
void runGlobalTest(int, int, const char*, const char*);
template <typename T>
void runSharedMemTest(int, int, const char*, const char*);


int main() {
//  std::string storagePath = setupStoragePath(argc, argv);

  int blockSize = 256;
  int iterations = 1;
  
  std::string out1;
  std::string out2;
  // out1 = storagePath + std::string("outputAddFP32_1.csv");
  // out2 = storagePath + std::string("outputAddFP32_2.csv");
  
  // runTestGeneric<L2MemTest, float>(iterations, blockSize, "tmp1.csv", "tmp2.csv");

  // printf("---- beginning L1 Testing ----\n"); 
  // runL1Test<float>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");
  
  // printf("---- beginning L2 Testing ----\n"); 
  // runL2Test<float>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");
  
  // printf("---- beginning Global Memory Testing ----\n"); 
  // runGlobalTest<float>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");

  // printf("---- beginning Shared Memory Testing ----\n"); 
  // runSharedMemTest<float>(iterations, blockSize, "tmp1.csv", "tmp2.csv");
  // printf("---- test end ----\n");

  return 0;
}


// template <typename kernel, T>
// void runTestGeneric(int iterNum, int blockSize, const char* outputName1, 
//               const char* outputName2) 
// {
//   printf("Starting Kernel1\n");
//   kernel<T> test1(blockSize, iterNum);
//   TestRunner<kernel<T>> tester1(&test1, outputName1);
//   tester1.getGoodSample();
//   // tester1.dataToFile();
//   printf("Kernel 1 finished\n");
// }

template <typename T>
void runL1Test(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  L1MemTest<T> test1(blockSize, iterNum);
  TestRunner<L1MemTest<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
}

template <typename T>
void runL2Test(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  L2MemTest<T> test1(blockSize, iterNum);
  TestRunner<L2MemTest<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
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
}


template <typename T>
void runSharedMemTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  SharedMemTest<T> test1(blockSize, iterNum);
  TestRunner<SharedMemTest<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  // tester1.dataToFile();
  printf("Kernel 1 finished\n");
}
