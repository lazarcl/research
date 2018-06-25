#include "testFramework.cu"
#include "arithmeticTests.cu"
#include <string> 
#include <sys/stat.h>
#include "testHelpers.h"
#include <tuple>
#include <vector>



/*run command
nvcc runBasePowerTest1.cu -lnvidia-ml
*/


template <typename T>
void runSharedMemAddTest(int iterNum, int blockSize, float memRatio, 
            const char* outputName, float acceptableError) 
{
  MultKernel1TestSetSharedMem<T> test1(blockSize, iterNum);
  printf("  memRatio set to %f", memRatio);
  test1.setSharedMem(memRatio);
  TestRunner<MultKernel1TestSetSharedMem<T>> tester1(&test1, outputName, acceptableError);
  tester1.getGoodSample();
  tester1.dataToFile();
}


//given a kernel class and type to run base power with,return vector of tuples.
//KernelClass type already has the datatype of the class in it's own type: ex: AddFP32<float>
//Tuples are (blocksPerSM, avgPower, elapsedTime) for each basePower run
template <typename KernelClass>
vector< tuple<int,float,float> > basePowerTest1_SpecifyKernel() {
  int blockSize = 256;
  int addIter = config_t.basePow1_iter;
  float acceptableError = 1000; //set large so it has no affect 

  vector<tuple<int, float, float>> runsVector;

  printf("---- beginning kernel's runs of the 1st approach to base power measuring ----"); 
  for (int blckPerSM = 1; blckPerSM <= 8; blckPerSM++) {

    //what percent of shared mem for each thread to request
    float memRatio = 1.0f/((float)blckPerSM) - 0.02f;

    //if foobar is a string obj. get const char* with: foobar.c_str()
    std::string pathName = folderPath + std::string("/outputBlksPerSM_");
    std::string fileType = ".csv";
    std::string numStr = std::to_string(blckPerSM);

    printf("---- beginning run #%d ----\n", blckPerSM); 
    KernelClass test1(blockSize, iterNum);
    printf("  memRatio set to %f", memRatio);
    test1.setSharedMem(memRatio);
    TestRunner<KernelClass> tester1(&test1, outputName, acceptableError);
    tester1.getGoodSample();
    
    runsVector.push_back( tuple<int, float, float>(blckPerSM, (float)tester1.getPowerAvg(), tester1.getElapsedTime()));
    
    printf("---- test end ----\n");
    return runsVector;
  }


  KernelClass<T> test1()
}

int main(int argc, char *argv[]) {
  int blockSize = 256;
  int addIter = config_t.basePow1_iter;
  float acceptableError = 1000; //set large so it has no affect 
  
  std::string folderPath = setupStoragePath(argc, argv);

  printf("---- beginning runs of the 1st approach to base power measuring. Storing at '%s' ----\n", folderPath.c_str()); 
  for (int blckPerSM = 1; blckPerSM <= 8; blckPerSM++) {

    //what percent of shared mem for each thread to request
    float memRatio = 1.0f/((float)blckPerSM) - 0.02f;

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











