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
std::vector< std::tuple<int,float,float> > basePowerTest1_SpecifyKernel() {
  int blockSize = 256;
  int iters = config_t.basePow1_iter;
  float acceptableError = 1000; //set large so it has no affect 

  std::vector<std::tuple<int, float, float>> runsVector;

  printf("---- beginning kernel's runs of the 1st approach to base power measuring ----\n"); 
  for (int sampleDepth = 1; sampleDepth <= 4; sampleDepth++) {
    for (int blckPerSM = 1; blckPerSM <= 6; blckPerSM++) {

      //what percent of shared mem for each thread to request
      float memRatio = 1.0f/((float)blckPerSM) - 0.02f;


      printf("---- beginning run #%d ----\n", blckPerSM); 
      KernelClass test1(blockSize, iters);
      printf("  memRatio set to %f\n", memRatio);
      test1.setSharedMem(memRatio);
      TestRunner<KernelClass> tester1(&test1, "deleteMe.csv", acceptableError);
      tester1.getGoodSample();
      
      runsVector.push_back( std::tuple<int, float, float>(blckPerSM, (float)tester1.getPowerAvg(), tester1.getElapsedTime()));
      
      printf("---- test end ----\n");
    }
  }
  return runsVector;
}


void runClassicBPWithAddFP32(int argc, char* argv[]) {
  int blockSize = 256;
  int addIter = config_t.basePow1_iter;
  float acceptableError = 1000; //set large so it has no affect 
  
  std::string folderPath = setupStoragePath(argc, argv);

  printf("---- beginning runs of the 1st approach to base power measuring. Storing at '%s' ----\n", folderPath.c_str()); 
  for (int blckPerSM = 1; blckPerSM <= 7; blckPerSM++) {

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


void runBPWithAllKernels() {
  std::vector< std::tuple<int,float,float> > powData;
  powData = basePowerTest1_SpecifyKernel<AddKernel1TestSetSharedMem<float>>();
  basePowVectorToFile(powData, "testing/basePow1_addFloat.csv");
  powData = basePowerTest1_SpecifyKernel<AddKernel1TestSetSharedMem<double>>();
  basePowVectorToFile(powData, "testing/basePow1_addDouble.csv");
  powData = basePowerTest1_SpecifyKernel<AddKernel1TestSetSharedMem<int>>();
  basePowVectorToFile(powData, "testing/basePow1_addInt.csv");
//
//  powData = basePowerTest1_SpecifyKernel<MultKernel1TestSetSharedMem<int>>();
//  basePowVectorToFile(powData, "testing/basePow1_multInt.csv");
//  powData = basePowerTest1_SpecifyKernel<MultKernel1TestSetSharedMem<float>>();
//  basePowVectorToFile(powData, "testing/basePow1_multFloat.csv");
//  powData = basePowerTest1_SpecifyKernel<MultKernel1TestSetSharedMem<double>>();
//  basePowVectorToFile(powData, "testing/basePow1_multDouble.csv");

//  powData = basePowerTest1_SpecifyKernel<FMAKernel1TestSetSharedMem<float>>();
//  basePowVectorToFile(powData, "testing/basePow1_fmaFloat.csv");
//  powData = basePowerTest1_SpecifyKernel<FMAKernel1TestSetSharedMem<double>>();
//  basePowVectorToFile(powData, "testing/basePow1_fmaDouble.csv");
}


int main(int argc, char *argv[]) {
  runBPWithAllKernels();
  
  return 0;
}











