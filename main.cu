#include "testFramework.cu"


/*run command
nvcc main.cu -L/usr/lib64/nvidia -lnvidia-ml -I/usr/local/cuda-7.0/samples/common/inc/ -I/nvmlPower.cpp
*/


template <typename T>
void runAddTest(int iterNum, int blockSize, const char* outputName1, 
              const char* outputName2) 
{
  printf("Starting Alg1\n");
  AdditionFP_1<T> test1(blockSize, iterNum);
  TestRunner<AdditionFP_1<T>> tester1(&test1, outputName1);
  tester1.getGoodSample();
  tester1.dataToFile();

  printf("Alg 1 finished\n");

  printf("Starting Alg2\n");
  AdditionFP_2<T> test2(blockSize, iterNum);
  TestRunner<AdditionFP_2<T>> tester2(&test2, outputName2);
  tester2.getGoodSample();
  tester2.dataToFile();

  printf("Alg 2 finished\n");
}


int main() {
  printf("---- beginning FP32 Add Testing ----\n"); 
  runAddTest<float>(1000000, 256, "data/outputAddFP32_1.txt", "data/outputAddFP32_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning FP64 Add Testing ----\n");
  runAddTest<double>(1000000, 256, "data/outputAddFP64_1.txt", "data/outputAddFP64_2.txt");
  printf("---- test end ----\n");

  printf("---- beginning Int32 Add Testing ---\n");
  runAddTest<int>(1000000, 256, "data/outputAddInt32_1.txt", "data/outputAddInt32_2.txt");
  printf("---- test end ----\n");



  return 0;
}
