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
                       "runBasePowerTest2.cu":"basePowerTest2.out"}


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

  tStart = time.time()
  while True:
      out = popen.stdout.read(1)
      if time.time() - tStart > 5:
        break
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

#given list of files to compile, ensure they all compile with no output
#name the executabe outputs according to testExecutableNames dictionary
def compileAll(testFiles):
  print("compiling tests...")
  for test in testFiles:
    print("  compiling", test + "...", end='')
    outName = testExecutableNames[test]
    exitStatus = runCommandPrintOutput( ("nvcc", test, "-lnvidia-ml", "-o", outName) )
    if exitStatus != 0:
      print(test, "didn't compile cleanly. Quitting for debug")
      exit(1)
    print("DONE")
  print("DONE compiling")

#given executable name and storage path, run the executable
#handle unsucessful exit by retrying. 
#If still unsucessful: return 1 
#if successful: return 0
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
      return 1
  print("END TEST: '" + str(command) + "'")
  print("")
  return 0

#given list of storage paths, run each test once for each storagePath
def runTestsForDirs(testFiles, storagePaths):
  for path in storagePaths:
    for test in testFiles:
      testExec = testExecutableNames[test]
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

  tests = ["runArithmeticTests.cu", "runBasePowerTest1.cu", "runBasePowerTest2.cu"]
  compileAll(tests)
 
  runTestsForDirs(tests, dirList)
#  command = ("./arithmeticTest.out", dirList[0])
#  runCommand(command)
  # runExec(testExecutableNames[tests[0]], dirList[0])










