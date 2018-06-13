

#ifndef TESTHELPERS_H
#define TESTHELPERS_H


#include <string>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>


//return the path to store results at. If path is not an argument,
//then create a new directory with a unique name and return that.
std::string setupStoragePath(int argc, char *argv[]);

//used by setupStoragePath to create a folder in './data' using the
// filename and unique suffix
std::string makeNewPath(char *argv[]);


#endif