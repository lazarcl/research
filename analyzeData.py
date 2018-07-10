from testScripts.arithmeticTestPlotter import *
from testScripts.basePowTestPlotter import *
from testScripts.basePowVarCalculator import *
from testScripts.testSpreadCalculator import *
import testScripts.dataLoader as dataLoader
from testScripts.mathHelpers import *
import testScripts.analysisConfig as analysisConfig
from testScripts.basePowForKernels import *
import pathlib
import os
import math


def calculateBasePower(rootPath, saveDir, dataDirs):
  savePath = saveDir + analysisConfig.basePowerAnalysisFilename
  print("Calculating base power from approach 1")
  obj = BasePowVarCalculator(dataDirs, [1,2,3,4,5,6], analysisConfig.basePower1GenericName)
  obj.calcBasePow()
  # print("Results for basePow 1:")
  # obj.printBasePowers()

  print("Calculating base power from approach 2")
  obj2 = BasePowVarCalculator(dataDirs, [3,4,5], analysisConfig.basePower2GenericName)
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
  # testNames = ["AddFP32", "AddFP64", "AddInt32", "FMAFP32", "FMAFP64", "MultFP32", "MultFP64", "MultInt32"]
  testNames = analysisConfig.arithTestNames
  pdfName = analysisConfig.arithGenericGraphPdfName

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

  # arithOutputNames = ['outputAddFP32_1.csv', 'outputAddFP64_1.csv', 'outputAddInt32_1.csv', \
  #     'outputFMAFP32_1.csv', 'outputFMAFP64_1.csv', 'outputMultFP32_1.csv', \
  #     'outputMultFP64_1.csv', 'outputMultInt32_1.csv', 'outputAddFP32_2.csv', \
  #     'outputAddFP64_2.csv', 'outputAddInt32_2.csv', 'outputFMAFP32_2.csv', \
  #     'outputFMAFP64_2.csv', 'outputMultFP32_2.csv', 'outputMultFP64_2.csv', \
  #     'outputMultInt32_2.csv']

  
  testSpreads = TestSpreadCalculator(analysisConfig.arithOutputFiles, rootPath, ignoreDirectories)
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

  controlTime = timeDict[controlFile]
  testTime = timeDict[testFile]
  controlEnergy = energyDict[controlFile]
  testEnergy = energyDict[testFile]

  controlBasePowEnergy = multiplyIndVar(controlTime, basePow)
  testBasePowEnergy = multiplyIndVar(testTime, basePow)

  controlOpEnergy = subIndVar(controlEnergy, controlBasePowEnergy)
  testOpEnergy = subIndVar(testEnergy, testBasePowEnergy)
  marginalEnergy = subIndVar(testOpEnergy, controlOpEnergy)

  controlOpCount, controlThreadCount = dataLoader.getOpAndThreadCountFromFile(glob.glob(analysisConfig.pathDict["baseDir"]+"run*/"+controlFile)[0])
  testOpCount, testThreadCount = dataLoader.getOpAndThreadCountFromFile(glob.glob(analysisConfig.pathDict["baseDir"]+"run*/"+testFile)[0])
  marginalOps = testOpCount*testThreadCount - controlOpCount*controlThreadCount

  energyPerOp, energyPerOpVar = multIndVarAndConst(marginalEnergy, float(1/marginalOps))
  energyPerOp, energyPerOpVar = multIndVarAndConst((energyPerOp, energyPerOpVar), float(10**12))

  resultDict["controlPow"] = tupleToRoundedStrings(varToPercent(powerDict[controlFile]))
  resultDict["controlTime"] = tupleToRoundedStrings(varToPercent(controlTime))
  resultDict["controlEnergy"] = tupleToRoundedStrings(varToPercent(controlEnergy))
  resultDict["testPow"] = tupleToRoundedStrings(varToPercent(powerDict[testFile]))
  resultDict["testTime"] = tupleToRoundedStrings(varToPercent(testTime))
  resultDict["testEnergy"] = tupleToRoundedStrings(varToPercent(testEnergy))
  resultDict["basePow"] = tupleToRoundedStrings(varToPercent(basePow))
  resultDict["marginalEnergy"] = tupleToRoundedStrings(varToPercent(marginalEnergy))
  resultDict["marginalOps"] = "{:.3e}".format(marginalOps)
  resultDict["energyPerOp"] = tupleToRoundedStrings(varToPercent((energyPerOp, energyPerOpVar)))

  return resultDict



#given two populated column dictionaries, return a populated latex table.
def makeTableFromCols(col1, col2, col1Name, col2Name):
  control = 'outputAddFP32_1.csv'
  test = 'outputAddFP32_2.csv'

  table = "\\begin{tabular}{|l|p{1.5in}|p{1.5in}|} \hline\n"
  
  table+= "Measurement  & "+col1Name+" & "+col2Name+"\\\ \hline\n"
  
  table+= "Control Kernel Power & " + col1["controlPow"][0] + "$\pm$" + col1["controlPow"][1] + "\% W & " + col1["controlPow"][0] + "$\pm$" + col1["controlPow"][1] + "\% W\\\ \hline\n"
  table+= "Control Kernel Time & "+ col1["controlTime"][0] + "$\pm$"+ col1["controlTime"][1]+"\% s & "+ col1["controlTime"][0] + "$\pm$"+ col1["controlTime"][1]+"\% s\\\ \hline\n"
  table+= "Control Kernel Energy & "+ col1["controlEnergy"][0] + "$\pm$"+ col1["controlEnergy"][1]+"\% J & "+ col1["controlEnergy"][0] + "$\pm$"+ col1["controlEnergy"][1]+"\% J\\\ \hline\n"
  
  table+= "Test Kernel Power & " + col1["testPow"][0] + "$\pm$" + col1["testPow"][1] + "\% W & " + col1["testPow"][0] + "$\pm$" + col1["testPow"][1] + "\% W\\\ \hline\n"
  table+= "Test Kernel Time & "+ col1["testTime"][0] + "$\pm$"+ col1["testTime"][1]+"\% s & "+ col1["testTime"][0] + "$\pm$"+ col1["testTime"][1]+"\% s\\\ \hline\n"
  table+= "Test Kernel Energy & "+ col1["testEnergy"][0] + "$\pm$"+ col1["testEnergy"][1]+"\% J & "+ col1["testEnergy"][0] + "$\pm$"+ col1["testEnergy"][1]+"\% J\\\ \hline\n"
  table+= "Base Power & "+ col1["basePow"][0] + "$\pm$"+ col1["basePow"][1]+"\% J & "+ col1["basePow"][0] + "$\pm$"+ col1["basePow"][1]+"\% J\\\ \hline\n"

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

  # testSpreadsObj2 = arithmeticTestSpreads("testRuns/p6000_second_set/")

  kernelBPAnalysis = BasePowForKernels("testRuns/k20_sixth_set_newKernels/run4", [1,2], analysisConfig.basePow2ResultFiles, 2)
  kernelBPAnalysis.calcBasePows()
  basePowResults = kernelBPAnalysis.getResults()
  # print(basePowResults)
  # quit(0)

  testSpreadsObj = arithmeticTestSpreads(rootPath)
  for name, (control, test) in analysisConfig.arithTestNamesToFiles.items():
  # for control, test in analysisConfig.arithOutputPairs:
    # basePow = 36.0, 25
    col = makeTableColEntry(basePowResults[name][0][2:], testSpreadsObj, control, test)
    # col2 = makeTableColEntry(basePow, testSpreadsObj2, control, test)
    # print(str(col["energyPerOp"]))
    print("$"+name+"$\\\ \n"+makeTableFromCols(col, col, "K20", "K20"))
    # print("$"+name+"$\\\ \n"+makeTableFromCols(col, col2, "K20", "P6000"))



  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders in", rootPath, "found" )


  # calculateBasePower(rootPath, saveDir, dataDirs)
  # graphBasePower(rootPath, saveDir, dataDirs)
  # graphArithmetic(ro otPath, saveDir, dataDirs)


if __name__ == "__main__":
  analyzeData()


  # print("Calculating base power from approach 1")
  # obj = BasePowForKernels("testing/bpTests", [1,2], analysisConfig.basePow1ResultFiles, 1)
  # obj = BasePowForKernels("testing/bpTests/", [1,2], {"addFP32":"basePow1_addFloat.csv"}, 1)

  # print("Calculating base power from approach 2")
  # obj = BasePowForKernels("testing/bpTests/", [1,2], analysisConfig.basePow2ResultFiles, 2)
  # obj = BasePowForKernels("testing/bpTests/", [1,2], {"addFP32":"basePow2_addFloat.csv"}, 2)
  # obj.calcBasePows()






