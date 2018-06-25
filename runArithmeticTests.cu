#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <sys/stat.h>
#include <string>
#include "testHelpers.h"

/*run command
nvcc runArithmeticTests.cu -lnvidia-ml
*/

template <typename T>
void runAddTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2);

template <typename T>
void runMultTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2);

template <typename T>
void runFMATest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2);


//optional argument to specify storage directory. Default is 'data/arithmeticTests'
int main(int argc, char *argv[]) {

  std::string storagePath = setupStoragePath(argc, argv);

  int blockSize = 256;
  
  std::string out1;
  std::string out2;
//  printf("---- beginning FP32 Add Testing ----\n"); 
//  out1 = storagePath + std::string("outputAddFP32_1.csv");
//  out2 = storagePath + std::string("outputAddFP32_2.csv");
//  runAddTest<float>(config_t.AddFP32_iter, blockSize, out1.c_str(), out2.c_str());
//  printf("---- test end ----\n");

  printf("---- beginning FP64 Add Testing ----\n");
  out1 = storagePath + std::string("outputAddFP64_1.csv");
  out2 = storagePath + std::string("outputAddFP64_2.csv");
  runAddTest<double>(config_t.AddFP64_iter, blockSize, out1.c_str(), out2.c_str());
  printf("---- test end ----\n");

  printf("---- beginning Int32 Add Testing ---\n");
  out1 = storagePath + std::string("outputAddInt32_1.csv");
  out2 = storagePath + std::string("outputAddInt32_2.csv");
  runAddTest<int>(config_t.AddInt32_iter, blockSize, out1.c_str(), out2.c_str());
  printf("---- test end ----\n");

//  printf("\n");
//  printf("---- beginning FP32 Mult Testing ----\n"); 
//  out1 = storagePath + std::string("outputMultFP32_1.csv");
//  out2 = storagePath + std::string("outputMultFP32_2.csv");
//  runMultTest<float>(config_t.MultFP32_iter, blockSize, out1.c_str(), out2.c_str());
//  printf("---- test end ----\n");
//
//  printf("---- beginning FP64 Mult Testing ----\n");
//  out1 = storagePath + std::string("outputMultFP64_1.csv");
//  out2 = storagePath + std::string("outputMultFP64_2.csv");
//  runMultTest<double>(config_t.MultFP64_iter, blockSize, out1.c_str(), out2.c_str());
//  printf("---- test end ----\n");
//
//  printf("---- beginning Int32 Mult Testing ---\n");
//  out1 = storagePath + std::string("outputMultInt32_1.csv");
//  out2 = storagePath + std::string("outputMultInt32_2.csv");
//  runMultTest<int>(config_t.MultInt32_iter, blockSize, out1.c_str(), out2.c_str());
//  printf("---- test end ----\n");
//
//  printf("\n");
//  printf("---- beginning FP32 FMA Testing ----\n"); 
//  out1 = storagePath + std::string("outputFMAFP32_1.csv");
//  out2 = storagePath + std::string("outputFMAFP32_2.csv");
//  runFMATest<float>(config_t.FMAFP32_iter, blockSize, out1.c_str(), out2.c_str());
//  printf("---- test end ----\n");
//
//  printf("---- beginning FP64 FMA Testing ----\n");
//  out1 = storagePath + std::string("outputFMAFP64_1.csv");
//  out2 = storagePath + std::string("outputFMAFP64_2.csv");
//  runFMATest<double>(config_t.FMAFP64_iter, blockSize, out1.c_str(), out2.c_str());
//  printf("---- test end ----\n");

  return 0;
}



template <typename T>
void runAddTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  AddKernel1Test<T> test1(blockSize, iterNum);
  TestRunner<AddKernel1Test<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  tester1.dataToFile();

  printf("Kernel 1 finished\n");

  printf("Starting Kernel2\n");
  AddKernel2Test<T> test2(blockSize, iterNum);
  TestRunner<AddKernel2Test<T>> tester2(&test2, outputName2);
  tester2.getGoodSample();
  tester2.dataToFile();

  printf("Kernel 2 finished\n");
}

template <typename T>
void runMultTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  MultKernel1Test<T> test1(blockSize, iterNum);
  TestRunner<MultKernel1Test<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  tester1.dataToFile();

  printf("Kernel 1 finished\n");

  printf("Starting Kernel2\n");
  MultKernel2Test<T> test2(blockSize, iterNum);
  TestRunner<MultKernel2Test<T>> tester2(&test2, outputName2);
  tester2.getGoodSample();
  tester2.dataToFile();

  printf("Kernel 2 finished\n");
}

template <typename T>
void runFMATest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Kernel1\n");
  FmaKernel1Test<T> test1(blockSize, iterNum);
  TestRunner<FmaKernel1Test<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  tester1.dataToFile();

  printf("Kernel 1 finished\n");

  printf("Starting Kernel2\n");
  FmaKernel2Test<T> test2(blockSize, iterNum);
  TestRunner<FmaKernel2Test<T>> tester2(&test2, outputName2);
  tester2.getGoodSample();
  tester2.dataToFile();

  printf("Kernel 2 finished\n");
}






