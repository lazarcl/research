import subprocess
import pathlib


'''
TESTS TO RUN:
  runArithmeticTests.cu
  runBasePowerTest1.cu
  runBasePowerTest2.cu


MAKE SURE ALL RUNS TAKE AN OPTIONAL DIRECTORY PATH
  IF PROVIDED, THEN PLACE NORMAL OUTPUT IN THAT PATH

'''

testExecutableNames = {"runArithmeticTests.cu":"arithmeticTest.out", \
                       "runBasePowerTest1.cu":"basePowerTest1.out", \
                       "runBasePowerTest2.cu":"basePowerTest2.out"}

# what folders should each set of runs be saved to
listOfRunPaths = ["run1/", "run2/", "run3/"]

tests = ["runArithmeticTests.cu", "runBasePowerTest1.cu", "runBasePowerTest2.cu"]

#make sure all directories in givenlist exist. If not, create them
def makeDirs(dirList):
  for path in dirList:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

#run command and print the subprocesses output in real time
#return the exit status of the subprocess
def runCommandPrintOutput(command):
  popen = subprocess.Popen(command, stdout=subprocess.PIPE)

  for line in iter(popen.stdout.readline, b''):
    print(line)
  popen.stdout.close()
  popen.wait()

  return popen.returncode

#given list of files to compile, ensure they all compile with no output
#name the executabe outputs according to testExecutableNames dictionary
def compileAll(testFiles):
  for test in testFiles:
    print("compiling", test, " ...")
    outName = testExecutableNames[test]
    exitStatus = runCommandPrintOutput( ("nvcc", test, "-lnvidia-ml", "-o", outName) )
    if exitStatus != 0:
      print(test, "didn't compile cleanly. Quitting before any runs for debug")
      exit(1)

#given executable name and storage path, run the executable
#handle unsucessful exit by retrying. 
#If still unsucessful: return 1 
#if successful: return 0
def runExec(execName, storagePath):
  command = ("./"+execName+" "+storagePath)
  exitStatus = runCommandPrintOutput(command)
  if exitStatus != 1:
    print(execName, "at", storagePath, "unsucessful. Retrying...")
    command = ("./"+execName+" "+storagePath)
    exitStatus = runCommandPrintOutput(command)
    if exitStatus != 0:
      print(execName, "at", storagePath, "failed for the seccond time")
      return 1
  return 0

#given list of storage paths, run each test once for each storagePath
def runTestsForDirs(storagePaths):
  for path in storagePaths:
    for testFile, testExec in testExecutableNames.items():
      exitStatus = runExec(testExec, path)
      if exitStatus != 0:
        print("Quitting because", testExec, "failed twice")
        exit(1)

#run the given command and return the processes output
def runCommand(command):
  popen = subprocess.Popen(command, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()
  return output


if __name__ == "__main__":
  dirList = ["testRuns/run1", "testRuns/run2"]
  makeDirs(dirList)

  testFile = ["runArithmeticTests.cu"]
  compileAll(testFile)

  runExec(testExecutableNames[testFile[0]], dirList[0])










