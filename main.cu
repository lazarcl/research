#include "testFramework.cu"
#include "arithmeticTests.cu"


/*run command
nvcc main.cu -L/usr/lib64/nvidia -lnvidia-ml -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp
*/


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


int main() {
  int addIter = 5000000;
  printf("---- beginning FP32 Add Testing ----\n"); 
  runAddTest<float>(multIter, 256, "data/outputAddFP32_1.txt", "data/outputAddFP32_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning FP64 Add Testing ----\n");
  runAddTest<double>(multIter, 256, "data/outputAddFP64_1.txt", "data/outputAddFP64_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning Int32 Add Testing ---\n");
  runAddTest<int>(multIter, 256, "data/outputAddInt32_1.txt", "data/outputAddInt32_2.txt");
  printf("---- test end ----\n");

  printf("\n");
  int multIter = 4000000;
  printf("---- beginning FP32 Mult Testing ----\n"); 
  runMultTest<float>(multIter, 256, "data/outputMultFP32_1.txt", "data/outputMultFP32_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning FP64 Mult Testing ----\n");
  runMultTest<double>(multIter, 256, "data/outputMultFP64_1.txt", "data/outputMultFP64_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning Int32 Mult Testing ---\n");
  runMultTest<int>(multIter, 256, "data/outputMultInt32_1.txt", "data/outputMultInt32_2.txt");
  printf("---- test end ----\n");

  printf("\n");
  int fmaIter = 100000;
  printf("---- beginning FP32 FMA Testing ----\n"); 
  runFMATest<float>(fmaIter, 256, "data/outputFMAFP32_1.txt", "data/outputFMAFP32_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning FP64 FMA Testing ----\n");
  runFMATest<double>(fmaIter, 256, "data/outputFMAFP64_1.txt", "data/outputFMAFP64_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning Int32 FMA Testing ---\n");
  runFMATest<int>(fmaIter, 256, "data/outputFMAInt32_1.txt", "data/outputFMAInt32_2.txt");
  printf("---- test end ----\n");



  return 0;
}
