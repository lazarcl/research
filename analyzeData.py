from testScripts.arithmeticTestPlotter import *
from testScripts.basePowTestPlotter import *
from testScripts.basePowVarCalculator import *
from testScripts.testSpreadCalculator import *
import testScripts.dataLoader as dataLoader
from testScripts.mathHelpers import *
import testScripts.analysisConfig as analysisConfig
import pathlib
import os
import math


def calculateBasePower(rootPath, saveDir, dataDirs):
  savePath = saveDir + "basePowerResults.txt"
  print("Calculating base power from approach 1")
  obj = BasePowVarCalculator(dataDirs, [1,2,3,4], "outputBlksPerSM_")
  obj.calcBasePow()
  # print("Results for basePow 1:")
  # obj.printBasePowers()

  print("Calculating base power from approach 2")
  obj2 = BasePowVarCalculator(dataDirs, [2,3,4], "outputBlockScalar_")
  obj2.calcBasePow()
  # print("Results for basePow 2:")
  # obj2.printBasePowers()

  print("Writing base power energy results to:", savePath + "\n")
  writeBasePowers(obj.getBasePowers(), obj2.getBasePowers(), savePath)


def graphBasePower(rootPath, saveDir, dataDirs):
  obj = BasePowTestPlotter(saveDir, dataDirs, analysisConfig.graphDict["graphHeight"])
  obj.makeBasePow2Graphs()
  obj.makeBasePow1Graphs()


def graphArithmetic(rootPath, saveDir, dataDirs):
  testNames = ["AddFP32", "AddFP64", "AddInt32", "FMAFP32", "FMAFP64", "MultFP32", "MultFP64", "MultInt32"]
  pdfName = "arithmeticGraphs_"

  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders found in", rootPath)

  obj = ArithmeticTestPlotter(rootPath, saveDir, dataDirs, pdfName, testNames,\
                              graphHeight=analysisConfig.graphDict["graphHeight"])
  obj.graphAndSaveData()


#given a path, look for all arithmetic-test output files in it's subdirectories.
#if there are directories to not search in, then add that directory
#  as a string to ignoreDirectories param
#Return: TestSpreadCalculator object to grab result dictionaries from
def arithmeticTestSpreads(rootPath, ignoreDirectories=[]):

  arithOutputNames = ['outputAddFP32_1.csv', 'outputAddFP64_1.csv', 'outputAddInt32_1.csv', \
      'outputFMAFP32_1.csv', 'outputFMAFP64_1.csv', 'outputMultFP32_1.csv', \
      'outputMultFP64_1.csv', 'outputMultInt32_1.csv', 'outputAddFP32_2.csv', \
      'outputAddFP64_2.csv', 'outputAddInt32_2.csv', 'outputFMAFP32_2.csv', \
      'outputFMAFP64_2.csv', 'outputMultFP32_2.csv', 'outputMultFP64_2.csv', \
      'outputMultInt32_2.csv']

  
  testSpreads = TestSpreadCalculator(arithOutputNames, rootPath, ignoreDirectories)
  testSpreads.findRuntimeSpreads()
  testSpreads.findPowerSpreads()
  testSpreads.findEnergySpreadsOfResults()
  # print("runtime spread results:")
  # testSpreads.printRuntimeSpreads()
  # print("power spread results:")
  # testSpreads.printPowerSpreads()
  # print("energy spread results:")
  # testSpreads.printEnergySpreads()

  return testSpreads




def makeTableColEntry(basePow, spreadObj, controlFile, testFile):
  resultDict = {}

  timeDict = spreadObj.getRuntimeSpreadDict()
  powerDict = spreadObj.getPowerSpreadDict()
  energyDict = spreadObj.getEnergySpreadDict()

  #store tuples in dict
  resultDict["controlPow"] = tupleToStringsRounding(powerDict[controlFile])
  resultDict["testPow"] = tupleToStringsRounding(powerDict[testFile])
  
  controlTime = timeDict[controlFile]
  testTime = timeDict[testFile]
  resultDict["controlTime"] = tupleToStringsRounding(controlTime)
  resultDict["testTime"] = tupleToStringsRounding(testTime)

  controlEnergy = energyDict[controlFile]
  testEnergy = energyDict[testFile]
  resultDict["controlEnergy"] = tupleToStringsRounding(controlEnergy)
  resultDict["testEnergy"] = tupleToStringsRounding(testEnergy)

  controlBasePowEnergy = multiplyIndVar(controlTime, basePow)
  testBasePowEnergy = multiplyIndVar(testTime, basePow)

  controlOpEnergy = controlEnergy[0] - controlBasePowEnergy[0], controlEnergy[1] - controlBasePowEnergy[1] #
  testOpEnergy = testEnergy[0] - testBasePowEnergy[0], testEnergy[1] - testBasePowEnergy[1]#

  marginalEnergy = testOpEnergy[0] - controlOpEnergy[0], testOpEnergy[1] + controlOpEnergy[1]#
  resultDict["marginalEnergy"] = tupleToStringsRounding(marginalEnergy)

  controlOpCount, controlThreadCount = dataLoader.getOpAndThreadCountFromFile(glob.glob(analysisConfig.pathDict["baseDir"]+"run*/"+controlFile)[0])
  testOpCount, testThreadCount = dataLoader.getOpAndThreadCountFromFile(glob.glob(analysisConfig.pathDict["baseDir"]+"run*/"+testFile)[0])
  marginalOps = testOpCount*testThreadCount - controlOpCount*controlThreadCount
  resultDict["marginalOps"] = "{:.3e}".format(marginalOps)

  energyPerOp = marginalEnergy[0] / marginalOps
  # energyPerOpVar = marginalEnergy[1] * (1/marginalOps)**2
  energyPerOpVar =  (1/marginalOps)**2 * marginalEnergy[1]

  resultDict["energyPerOp"] = tupleToStringsRounding((energyPerOp*10**12, energyPerOpVar))

  return resultDict



#given two populated column dictionaries, return a populated latex table.
def makeTableFromCols(col1, col2):
  control = 'outputAddFP32_1.csv'
  test = 'outputAddFP32_2.csv'

  table = "\\begin{tabular}{|l|p{1.5in}|p{1.5in}|} \hline\n"
  
  table+= "Measurement  & K20 & P100\\\ \hline\n"
  
  table+= "Control Kernel Power & " + col1["controlPow"][0] + "$\pm$" + col1["controlPow"][1] + "\% W & " + col1["controlPow"][0] + "$\pm$" + col1["controlPow"][1] + "\% W\\\ \hline\n"
  table+= "Control Kernel Time & "+ col1["controlTime"][0] + "$\pm$"+ col1["controlTime"][1]+"\% s & "+ col1["controlTime"][0] + "$\pm$"+ col1["controlTime"][1]+"\% s\\\ \hline\n"
  table+= "Control Kernel Energy & "+ col1["controlEnergy"][0] + "$\pm$"+ col1["controlEnergy"][1]+"\% J & "+ col1["controlEnergy"][0] + "$\pm$"+ col1["controlEnergy"][1]+"\% J\\\ \hline\n"
  
  table+= "Test Kernel Power & " + col1["testPow"][0] + "$\pm$" + col1["testPow"][1] + "\% W & " + col1["testPow"][0] + "$\pm$" + col1["testPow"][1] + "\% W\\\ \hline\n"
  table+= "Test Kernel Time & "+ col1["testTime"][0] + "$\pm$"+ col1["testTime"][1]+"\% s & "+ col1["testTime"][0] + "$\pm$"+ col1["testTime"][1]+"\% s\\\ \hline\n"
  table+= "Test Kernel Energy & "+ col1["testEnergy"][0] + "$\pm$"+ col1["testEnergy"][1]+"\% J & "+ col1["testEnergy"][0] + "$\pm$"+ col1["testEnergy"][1]+"\% J\\\ \hline\n"

  table+= "Test Kernel Marginal Energy & " + col1["marginalEnergy"][0] + "$\pm$" + col1["marginalEnergy"][1] + "\% J & " + col1["marginalEnergy"][0] + "$\pm$" + col1["marginalEnergy"][1] + "\% J\\\ \hline\n"

  table+= "Marginal Operations Per Kernel & " + col1["marginalOps"] + "& " + col1["marginalOps"] + " \\\ \hline\n"
  table+= "Energy Per Operation & " + col1["energyPerOp"][0] + "$\pm$" + col1["energyPerOp"][1] + "\% pJ & " + col1["energyPerOp"][0] + "$\pm$" + col1["energyPerOp"][1] + "\% pJ\\\ \hline\n"
  
  table+= "\end{tabular}\n"

  return table


def analyzeData():
  rootPath = analysisConfig.pathDict["baseDir"]
  saveDir = rootPath + analysisConfig.pathDict["saveDir"]
  dataDirs = glob.glob(rootPath + analysisConfig.pathDict["dataDirs"])

  pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True) 
  print("Analyzing data from directory: '" + rootPath + "'")
  print("   Saving analysis output in: '" + saveDir + "'")
  print("   Looking in directories named like: '" + analysisConfig.pathDict["dataDirs"] + "'\n")
  # # rootPath = "testRuns/k20_second_set/"
  # # saveDir = rootPath + "analysis/"
  # # dataDirs = glob.glob(rootPath + "run*/")

  testSpreadsObj = arithmeticTestSpreads(rootPath)
  # table = makeTableFromCols(col1, col2)

  arithOutputNames = [('outputAddFP32_1.csv', 'outputAddFP32_2.csv'),  \
                      ('outputAddFP64_1.csv', 'outputAddFP64_2.csv'), \
                      ('outputAddInt32_1.csv', 'outputAddInt32_2.csv'), \
                      ('outputFMAFP32_1.csv', 'outputFMAFP32_2.csv'), \
                      ('outputFMAFP64_1.csv', 'outputFMAFP64_2.csv'),  \
                      ('outputMultFP32_1.csv', 'outputMultFP32_2.csv'),\
                      ('outputMultFP64_1.csv', 'outputMultFP64_2.csv'), \
                      ('outputMultInt32_1.csv', 'outputMultInt32_2.csv')  \
                     ]

  for control, test in arithOutputNames:
    print("Results for", control, test)
    col = makeTableColEntry((36.5,0.02), testSpreadsObj, control, test)
    # print(str(col["energyPerOp"]), "\n")
    print(makeTableFromCols(col, col))

  # # for a, b in col.items():
  # #   print(str(a), str(b))

  exit(0)


  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders in", rootPath, "found" )


  calculateBasePower(rootPath, saveDir, dataDirs)
  graphBasePower(rootPath, saveDir, dataDirs)
  graphArithmetic(rootPath, saveDir, dataDirs)


if __name__ == "__main__":
  analyzeData()





