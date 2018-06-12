#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <sys/stat.h>


/*run command
nvcc runArithmeticTests.cu -lnvidia-ml
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

std::string setupStoragePath(int argc, char *argv[]) {
  std::string storagePath;
  if (argc > 2) {
    printf("Too many arguments for %s. Expected max of 1 optional argument: storagePath\n", argv[0]);
  } else if (argc == 2) {
    storagePath = std::string(argv[1]);
    if (storagePath.back() != '/') {
      storagePath += std::string("/");
    } 
  } else if (argc == 1) {
    storagePath = std::string("data/arithmeticTests/");
  }

  struct stat sb;
  printf("storing data at: '%s'\n", storagePath.c_str());
  if (stat(pathname, &sb) != 0) {
    printf("Storage Path directory: '%s', does not exist\n", storagePath.c_str());
    if ( 0 == mkdir(storagePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
      printf("Storage path directory '%s' created\n", storagePath);
    } else {
      printf("Storeage path not resolved, exiting as unsucessful\n");
      exit(1);
    }
  }

  return storagePath;

}

//optional argument to specify storage directory. Default is 'data/arithmeticTests'
int main(int argc, char *argv[]) {

  std::string storagePath = setupStoragePath(argc, argv);

  int blockSize = 256;
  
  int addIter = 2000000;
  std::string out1;
  std::string out2;
  printf("---- beginning FP32 Add Testing ----\n"); 
  out1 = storagePath + std::string("outputAddFP32_1.csv");
  out2 = storagePath + std::string("outputAddFP32_2.csv");
  runAddTest<float>(addIter, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("---- beginning FP64 Add Testing ----\n");
  out1 = storagePath + std::string("outputAddFP64_1.csv");
  out2 = storagePath + std::string("outputAddFP64_2.csv");
  runAddTest<double>(addIter/3, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("---- beginning Int32 Add Testing ---\n");
  out1 = storagePath + std::string("outputAddInt32_1.csv");
  out2 = storagePath + std::string("outputAddInt32_2.csv");
  runAddTest<int>(addIter, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("\n");
  int multIter = 2000000;
  printf("---- beginning FP32 Mult Testing ----\n"); 
  out1 = storagePath + std::string("outputMultFP32_1.csv");
  out2 = storagePath + std::string("outputMultFP32_2.csv");
  runMultTest<float>(multIter, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("---- beginning FP64 Mult Testing ----\n");
  out1 = storagePath + std::string("outputMultFP64_1.csv");
  out2 = storagePath + std::string("outputMultFP64_2.csv");
  runMultTest<double>(multIter/3, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("---- beginning Int32 Mult Testing ---\n");
  out1 = storagePath + std::string("outputMultInt32_1.csv");
  out2 = storagePath + std::string("outputMultInt32_2.csv");
  runMultTest<int>(multIter, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("\n");
  int fmaIter = 800000;
  printf("---- beginning FP32 FMA Testing ----\n"); 
  out1 = storagePath + std::string("outputFMAFP32_1.csv");
  out2 = storagePath + std::string("outputFMAFP32_2.csv");
  runFMATest<float>(fmaIter*2, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  printf("---- beginning FP64 FMA Testing ----\n");
  out1 = storagePath + std::string("outputFMAFP64_1.csv");
  out2 = storagePath + std::string("outputFMAFP64_2.csv");
  runFMATest<double>(fmaIter, blockSize, out1.std::c_str(), out2.std::c_str());
  printf("---- test end ----\n");

  return 0;
}
