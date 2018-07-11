#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include <tuple>
#include <vector>



/*run command
nvcc runBasePowerTest2.cu -lnvidia-ml
*/


template <typename T>
void runAddTest(int iterNum, int blockSize, int blockSizeScalar, 
            const char* outputName, float acceptableError) 
{
  AddKernel1Test<T> test1(blockSize, iterNum, blockSizeScalar);
  TestRunner<AddKernel1Test<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}

template <typename T>
void runMultTest(int iterNum, int blockSize, int blockSizeScalar,
		const char* outputName, float acceptableError)
{
  MultKernel1TestNonVolatile<T> test1(blockSize, iterNum, blockSizeScalar);
  TestRunner<MultKernel1TestNonVolatile<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}

template <typename T>
void runAddTestVolatile(int iterNum, int blockSize, int blockSizeScalar, 
            const char* outputName, float acceptableError) 
{
  AddKernel1TestVolatile<T> test1(blockSize, iterNum, blockSizeScalar);
  TestRunner<AddKernel1TestVolatile<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}

void runClassicBP2WithAddFP32(int argc, char* argv[]) {
  int blockSize = 256;
  int addIter = config_t.basePow2_iter;
  float acceptableError = 1000; //set large so it has no affect 
  
  std::string folderPath = setupStoragePath(argc, argv);

  printf("---- beginning runs of the 2nd approach to base power measuring. Storing at '%s' ----\n", folderPath.c_str()); 
  for (int blckScalr = 1; blckScalr <= 8; blckScalr++) {
    // char outName[100];   // array to hold the result.
    // strcpy(result,one); // copy string one into the result.
    // strcat(result,two); // append string two to the result.


    //if foobar is a string obj. get const char* with: foobar.c_str()
    std::string pathName = folderPath + std::string("/outputBlockScalar_");
    std::string fileType = ".csv";
    std::string numStr = std::to_string(blckScalr);
    const char *outName= (pathName + numStr + fileType).c_str();
    printf("---- beginning run #%d ----\n", blckScalr); 
    runMultTest<float>(addIter, blockSize, blckScalr,  
                       (pathName + numStr + fileType).c_str(), acceptableError);
    printf("---- test end ----\n");

  }
}

template <typename KernelClass>
std::vector< std::tuple<int, float, float> > basePowerTest2_SpecifyKernel() {
  int blockSize = 256;
  int iterNum = config_t.basePow2_iter;
  float acceptableError = 1000; //set large so it has no affect 
  
  std::vector<std::tuple<int, float, float>> runsVector;

  printf("---- beginning runs of the 2nd approach to base power measuring. ----\n"); 
  for (int blckScalr = 1; blckScalr <= 6; blckScalr++) {
    for (int testDepth = 1; testDepth <=1; testDepth++) {
  
      printf("---- beginning run #%d ----\n", blckScalr); 
      KernelClass test1(blockSize, iterNum, blckScalr);
      TestRunner<KernelClass> tester1(&test1, "deleteMe.csv", acceptableError);
      tester1.getGoodSample();
  
      runsVector.push_back( std::tuple<int, float, float>(blckScalr, (float)tester1.getPowerAvg(), tester1.getElapsedTime()));
  
      printf("---- test end ----\n");
  
    }
  }
  return runsVector;
}


void basePowVectorToFile(std::vector< std::tuple<int,float,float> > vec,  const char* fileName){
  FILE *fp = fopen(fileName, "w+");
  if (fp == NULL) {
    printf("Attempt at opening '%s' failed. Error: ", fileName);
    perror("");
    printf("Terminating...");
    exit(0);
  }
  fprintf(fp, "runID, avgPower, elapsedTime\n");
  
  for (int i = 0; i < vec.size(); i++){
    std::tuple<int,float,float> tup = vec[i];
    fprintf(fp, "%d, %.3lf, %.3lf\n", std::get<0>(tup), std::get<1>(tup)/1000.0, std::get<2>(tup)/1000.0);
  }
  fclose(fp);
}

void runBP2WithAllKernels(std::string storagePath) {
  std::vector< std::tuple<int,float,float> > powData;
  powData = basePowerTest2_SpecifyKernel<AddKernel1Test<float>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_addFloat.csv")).c_str());
  powData = basePowerTest2_SpecifyKernel<AddKernel1Test<double>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_addDouble.csv")).c_str());
  powData = basePowerTest2_SpecifyKernel<AddKernel1Test<int>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_addInt.csv")).c_str());

  powData = basePowerTest2_SpecifyKernel<MultKernel1Test<int>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_multInt.csv")).c_str());
  powData = basePowerTest2_SpecifyKernel<MultKernel1Test<float>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_multFloat.csv")).c_str());
  powData = basePowerTest2_SpecifyKernel<MultKernel1Test<double>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_multDouble.csv")).c_str());

  powData = basePowerTest2_SpecifyKernel<FmaKernel1Test<float>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_fmaFloat.csv")).c_str());
  powData = basePowerTest2_SpecifyKernel<FmaKernel1Test<double>>();
  basePowVectorToFile(powData, (storagePath + std::string("basePow2_fmaDouble.csv")).c_str());
}


int main(int argc, char *argv[]) {

  std::string storagePath = setupStoragePath(argc, argv);

  runBP2WithAllKernels(storagePath);

  return 0;
}











