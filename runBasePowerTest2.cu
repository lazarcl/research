#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 



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


int main() {
  int blockSize = 256;
  int addIter = 200000000;
  float acceptableError = 1000; //set large so it has no affect 
  
  printf("---- beginning runs of the 2nd approach to base power measuring ----\n"); 
  for (int blckScalr = 1; blckScalr <= 8; blckScalr++) {
    // char outName[100];   // array to hold the result.
    // strcpy(result,one); // copy string one into the result.
    // strcat(result,two); // append string two to the result.


    //if foobar is a string obj. get const char* with: foobar.c_str()
    std::string pathName = "data/basePow2/outputBlockScalar_";
    std::string fileType = ".txt";
    std::string numStr = std::to_string(blckScalr);
    const char *outName= (pathName + numStr + fileType).c_str();
    printf("---- beginning run %d ----\n", blckScalr); 
    runAddTest<float>(addIter, blockSize, blckScalr,  
                       (pathName + numStr + fileType).c_str(), acceptableError);
    printf("---- test end ----\n");

  }

  return 0;
}











