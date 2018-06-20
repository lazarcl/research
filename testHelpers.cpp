

#include "testHelpers.h"


std::string makeNewPath(char *argv[]) {

  //don't overwrite data. make a new directory
  std::string fileName = std::string(argv[0]);
  std::string name = fileName.substr(0, fileName.find(".", 0));
  std::string basePath = "data/" + name;
  std::string folderPath = basePath;
  int i = 1;
  while ( -1 == mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
    folderPath = basePath + std::string("_") + std::to_string(i);
    i++;
    if (i > 15) {
        printf("Error creating a directory for %s results \n", argv[0]);
        exit(1);
    }
  }
  return folderPath + std::string("/");
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
    storagePath = makeNewPath(argv);
  }

  struct stat sb;
  printf("storing data at: '%s'\n", storagePath.c_str());
  if (stat(storagePath.c_str(), &sb) != 0) {
    printf("Storage Path directory: '%s', does not exist\n", storagePath.c_str());
    if ( 0 == mkdir(storagePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
      printf("Storage path directory '%s' created\n", storagePath.c_str());
    } else {
      printf("Storeage path not resolved, exiting as unsucessful\n");
      exit(1);
    }
  }

  return storagePath;

}


