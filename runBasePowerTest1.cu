#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 
#include <sys/stat.h>



/*run command
nvcc runBasePowerTest1.cu -lnvidia-ml
*/


template <typename T>
void runSharedMemAddTest(int iterNum, int blockSize, int blckPerSM, 
            const char* outputName, float acceptableError) 
{
  AddKernel1TestSetSharedMem<T> test1(blockSize, iterNum);
  test1.setSharedMem(blckPerSM);
  TestRunner<AddKernel1TestSetSharedMem<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}


int main() {
  int blockSize = 256;
  int addIter = 200000000;
  float acceptableError = 1000; //set large so it has no affect 
  
  //don't overwrite data. make a new directory
  std::string basePath = "data/basePow1";
  std::string folderPath = basePath;
  int i = 1;
  while ( -1 == mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
    folderPath = basePath + std::string("_") + std::to_string(i);
    i++;
    if (i > 30) {
        printf("Error creating a directory for runBasePowerTest1.cu results \n");
        exit(1);
    }
  }

  printf("---- beginning runs of the 1st approach to base power measuring. Storing at '%s' ----\n", folderPath.c_str()); 
  for (int blckPerSM = 1, ; blckPerSM <= 8; blckPerSM++) {

    //what percent of shared mem for each thread to request
    float memRatio = 1/(float)blckPerSM - 0.02;

    //if foobar is a string obj. get const char* with: foobar.c_str()
    std::string pathName = folderPath + std::string("/outputBlksPerSM_");
    std::string fileType = ".csv";
    std::string numStr = std::to_string(blckPerSM);

    printf("---- beginning run #%d ----\n", blckPerSM); 
    runSharedMemAddTest<float>(addIter, blockSize, blckPerSM,  
                       (pathName + numStr + fileType).c_str(), acceptableError);
    printf("---- test end ----\n");

  }

  return 0;
}











