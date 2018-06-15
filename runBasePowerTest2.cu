#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include "config.cpp"



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
void runAddTestVolatile(int iterNum, int blockSize, int blockSizeScalar, 
            const char* outputName, float acceptableError) 
{
  AddKernel1TestVolatile<T> test1(blockSize, iterNum, blockSizeScalar);
  TestRunner<AddKernel1TestVolatile<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}


int main(int argc, char *argv[]) {
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
    runAddTestVolatile<float>(addIter, blockSize, blckScalr,  
                       (pathName + numStr + fileType).c_str(), acceptableError);
    printf("---- test end ----\n");

  }

  return 0;
}











