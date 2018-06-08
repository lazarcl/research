#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 



/*run command
nvcc runBasePowerTest2.cu -lnvidia-ml
*/


template <typename T>
void runAddTest(int iterNum, int blockSize, int blockSizeScalar, 
            const char* outputName) 
{
  AddKernel1Test<T> test1(blockSize, iterNum, blockSizeScalar);
  TestRunner<AddKernel1Test<T>> tester1(&test1, outputName);
  tester1.getGoodSample();
  tester1.dataToFile();
}


int main() {
  int blockSize = 256;
  int addIter = 2000000;
  
  printf("---- beginning runs of the 2nd approach to base power measuring ----\n"); 
  for (int blckScalr = 1; blckScalr <= 3; blckScalr++) {
    // char outName[100];   // array to hold the result.
    // strcpy(result,one); // copy string one into the result.
    // strcat(result,two); // append string two to the result.


    //if foobar is a string obj. get const char* with: foobar.c_str()
    std::string pathName = "data/basePow2/outputScalar"
    std::string scalrStr = std::to_string(blckScalr);
    std::string fileType = ".txt"
    char outName[]= (pathName + scalrStr + fileType).c_str()

    printf("---- beginning run %d ----\n", blckScalr); 
    runAddTest<float>(addIter, blockSize, outName);
    printf("---- test end ----\n");

  }

  return 0;
}











