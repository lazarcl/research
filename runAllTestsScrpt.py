import subprocess
import pathlib
import sys
import time


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
                       "runBasePowerTest2.cu":"basePowerTest2.out",
                       "runMemoryTests.cu":"memoryTests.out"}



#make sure all directories in givenlist exist. If not, create them
def makeDirs(dirList):
  print("creating save directories...", end='')
  for path in dirList:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
  print("DONE")

#run command and print the subprocesses output in real time
#return the exit status of the subprocess
def runCommandPrintOutput(command):
  popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

  while True:
      out = popen.stdout.read(1)
      if popen.poll() is not None:
          break
      if out != '':
          sys.stdout.write(out.decode("utf-8"))
          sys.stdout.flush()

  
#  for line in iter(popen.stdout.readline, b''):
#    print(line)
  popen.stdout.close()
  popen.wait()

  return popen.returncode

def runMakefile():
  print("compiling tests...")
  exitStatus = runCommandPrintOutput( ("make") )
  if exitStatus != 0:
    print("didn't compile cleanly. Quitting for debug")
    exit(1)
  print("DONE compiling")

#given executable name and storage path, run the executable
#handle unsucessful exit by retrying. 
#return number of failed attempts
def runExec(execName, storagePath):
  command = ("./"+execName, storagePath)
  print("")
  print("BEGINNING TEST: '" + str(command) + "'")
  exitStatus = runCommandPrintOutput(command)
  if exitStatus != 0:
    print(execName, "at", storagePath, "unsucessful. Retrying...")
    exitStatus = runCommandPrintOutput(command)
    if exitStatus != 0:
      print(execName, "at", storagePath, "failed for the seccond time")
      return 2
    return 1
  print("END TEST: '" + str(command) + "'")
  print("")
  return 0

#given list of storage paths, run each test once for each storagePath
def runTestsForDirs(testFiles, storagePaths):
  failedRuns = []
  for path in storagePaths:
    for test in testFiles:
      testExec = testExecutableNames[test]
      exitStatus = runExec(testExec, path)
      if exitStatus == 1:
        print("ERROR:", testExec, "failed to run once, but recovered")
      elif exitStatus == 2:
        print("ERROR:", testExec, "failed to run twice and didnt recover")
        failedRuns.append( (testExec, path) )

  print("All runs finished:")
  if len(failedRuns) == 0:
    print("  all runs sucessfully passed")
  elif len(failedRuns) > 0:
    print("  the following runs failed")
    for run in failedRuns:
      print("    " + run[0], "storing at path: '" + run[1] + "'")

#run the given command and return the processes output
def runCommand(command):
  popen = subprocess.Popen(command, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()
  return output


if __name__ == "__main__":

  # basePath = "testRuns/k20_seventh_set/"
  basePath = "testRuns/p6000_seventh_set/"

  #dir is a list of directorys. 
  # Each directory gets its own run of the specified data
  # ex: if dirList.size == 5, then complete all tests 5 times
  dirList = [basePath + "run" + str(i) for i in range(3,4)]
  makeDirs(dirList)

  runMakefile()

  tests = ["runArithmeticTests.cu", "runBasePowerTest1.cu", "runBasePowerTest2.cu", "runMemoryTests.cu"]
  # tests = ["runBasePowerTest1.cu", "runBasePowerTest2.cu"]
  #tests = ["runArithmeticTests.cu"]
  runTestsForDirs(tests, dirList)










