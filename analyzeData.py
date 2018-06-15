from testScripts.arithmeticTestPlotter import *
from testScripts.basePowTestPlotter import *
from testScripts.basePowVarCalculator import *
import pathlib


def calculateBasePower(rootPath, saveDir, dataDirs):
  savePath = saveDir + "basePowerResults.txt"
  print("Calculating base power from approach 1")
  obj = BasePowVarCalculator(dataDirs, [3,4,5], "outputBlksPerSM_")
  obj.calcBasePow()
  # print("Results for basePow 1:")
  # obj.printBasePowers()

  print("\nCalculating base power from approach 2")
  obj2 = BasePowVarCalculator(dataDirs, [3,4,5], "outputBlockScalar_")
  obj2.calcBasePow()
  # print("Results for basePow 2:")
  # obj2.printBasePowers()

  print("Writing base power energy results to:", savePath)
  writeBasePowers(obj.getBasePowers(), obj2.getBasePowers(), savePath)


def graphBasePower(rootPath, saveDir, dataDirs):
  obj = BasePowTestPlotter(saveDir, dataDirs)
  obj.makeBasePow2Graphs()
  obj.makeBasePow1Graphs()


def graphArithmetic(rootPath, saveDir, dataDirs):
  testNames = ["AddFP32", "AddFP64", "AddInt32", "FMAFP32", "FMAFP64", "MultFP32", "MultFP64", "MultInt32"]
  pdfName = "arithmeticGraphs_"

  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders found in", rootPath)

  obj = ArithmeticTestPlotter(rootPath, saveDir, dataDirs, pdfName, testNames)
  obj.graphAndSaveData()


if __name__ == "__main__":

  rootPath = "testRuns/p6000_second_set/"
  saveDir = rootPath + "analysis/"
  dataDirs = glob.glob(rootPath + "run*/")

  pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True) 

  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders in", basePath, "found" )


  calculateBasePower(rootPath, saveDir, dataDirs)
  graphBasePower(rootPath, saveDir, dataDirs)
  graphArithmetic(rootPath, saveDir, dataDirs)





