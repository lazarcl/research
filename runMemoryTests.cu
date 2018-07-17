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
  int iterations = 2000000;
  
  std::string out1;
  std::string out2;
  // out1 = storagePath + std::string("outputAddFP32_1.csv");
  // out2 = storagePath + std::string("outputAddFP32_2.csv");
  
  // runTestGeneric<L2MemReadTest1<float>>(iterations, blockSize, "data/outputL2ReadTest_1.csv");
  // runTestGeneric<L2MemReadTest2<float>>(iterations, blockSize, "data/outputL2ReadTest_2.csv");

  // runTestGeneric<L1MemTest1<float>>(iterations, blockSize, "data/outputL1ReadTest_1.csv");
  // runTestGeneric<L1MemTest2<float>>(iterations, blockSize, "data/outputL1ReadTest_2.csv");

  runTestGeneric<SharedMemReadTest1<float>>(iterations, blockSize, "data/outputSharedReadTest_1.csv");
  runTestGeneric<SharedMemReadTest2<float>>(iterations, blockSize, "data/outputSharedReadTest_2.csv");


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
  TestRunner<kernel> tester1(&test1, outputName);
  tester1.getGoodSample();
  tester1.dataToFile();
  printf("Kernel '%s' finished\n", outputName);
}



// template <typename kernel>
// void runL1Test(int iterNum, int blockSize, const char* outputName1, 
//               const char* outputName2) 
// {
//   printf("Starting Kernel1\n");
//   L1MemTest1<T> test1(blockSize, iterNum);
//   TestRunner<L1MemTest1<T>> tester1(&test1, outputName1);
//   tester1.getGoodSample();
//   // tester1.dataToFile();
//   printf("Kernel 1 finished\n");
// }
// 
// template <typename kernel>
// void runL2Test(int iterNum, int blockSize, const char* outputName1, 
//               const char* outputName2) 
// {
//   printf("Starting Kernel1\n");
//   L2MemTest1<T> test1(blockSize, iterNum);
//   TestRunner<L2MemTest1<T>> tester1(&test1, outputName1);
//   tester1.getGoodSample();
//   // tester1.dataToFile();
//   printf("Kernel 1 finished\n");
// }
// 
// 
// 
// template <typename kernel>
// void runGlobalTest(int iterNum, int blockSize, const char* outputName1, 
//               const char* outputName2) 
// {
//   printf("Starting Kernel1\n");
//   GlobalMemTest1<T> test1(blockSize, iterNum);
//   TestRunner<GlobalMemTest1<T>> tester1(&test1, outputName1);
//   tester1.getGoodSample();
//   // tester1.dataToFile();
//   printf("Kernel 1 finished\n");
// }
// 
// 
// template <typename kernel>
// void runSharedMemTest(int iterNum, int blockSize, const char* outputName1, 
//               const char* outputName2) 
// {
//   printf("Starting Kernel1\n");
//   SharedMemTest1<T> test1(blockSize, iterNum);
//   TestRunner<SharedMemTest1<T>> tester1(&test1, outputName1);
//   tester1.getGoodSample();
//   // tester1.dataToFile();
//   printf("Kernel 1 finished\n");
// }
