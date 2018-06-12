#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 
#include <sys/stat.h>



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


int main() {
  int blockSize = 256;
  int addIter = 200000000;
  float acceptableError = 1000; //set large so it has no affect 
  
  //don't overwrite data. make a new directory
  std::string basePath = "data/basePow2";
  std::string folderPath = basePath;
  int i = 1;
  while ( -1 == mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
    folderPath = basePath + std::string("_") + std::to_string(i);
    i++;
    if (i > 30) {
        printf("Error creating a directory for runBasePowerTest2.cu results \n");
        exit(1);
    }
  }

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











