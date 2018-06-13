#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"



/*run command
nvcc runBasePowerTest1.cu -lnvidia-ml
*/


template <typename T>
void runSharedMemAddTest(int iterNum, int blockSize, int memRatio, 
            const char* outputName, float acceptableError) 
{
  AddKernel1TestSetSharedMem<T> test1(blockSize, iterNum);
  test1.setSharedMem(memRatio);
  TestRunner<AddKernel1TestSetSharedMem<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}


int main(int argc, char *argv[]) {
  int blockSize = 256;
  int addIter = 4000000;
  float acceptableError = 1000; //set large so it has no affect 
  
  std::string folderPath = setupStoragePath(argc, argv);
  
  printf("---- beginning runs of the 1st approach to base power measuring. Storing at '%s' ----\n", folderPath.c_str()); 
  for (int blckPerSM = 1; blckPerSM <= 8; blckPerSM++) {

    //what percent of shared mem for each thread to request
    float memRatio = 1/(float)blckPerSM - 0.02;

    //if foobar is a string obj. get const char* with: foobar.c_str()
    std::string pathName = folderPath + std::string("/outputBlksPerSM_");
    std::string fileType = ".csv";
    std::string numStr = std::to_string(blckPerSM);

    printf("---- beginning run #%d ----\n", blckPerSM); 
    runSharedMemAddTest<float>(addIter, blockSize, memRatio,  
                       (pathName + numStr + fileType).c_str(), acceptableError);
    printf("---- test end ----\n");

  }

  return 0;
}










