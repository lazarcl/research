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
  # runIDsToUse = [1,2,3,4,5,6]
  runIDsToUse = [1,2,3,4,5]

  savePath = saveDir + analysisConfig.basePowerAnalysisFilename
  print("Calculating base power from approach 1")
  # obj = BasePowVarCalculator(dataDirs, runIDsToUse, analysisConfig.basePower1GenericName)
  obj = BasePowVarCalculator(dataDirs, runIDsToUse, analysisConfig.basePow2ResultFiles["AddFP32"])
  obj.calcBasePow()

  # print("Results for basePow 1:")
  # obj.printBasePowers()

  print("Calculating base power from approach 2")
  # obj2 = BasePowVarCalculator(dataDirs, runIDsToUse, analysisConfig.basePower2GenericName)
  obj2 = BasePowVarCalculator(dataDirs, runIDsToUse, analysisConfig.basePow1ResultFiles["AddFP32"])
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

def graphMemory(rootPath, saveDir, dataDirs):
  testNames = analysisConfig.memoryTestNames
  pdfName = analysisConfig.memoryGenericGraphPdfName

  if len(dataDirs) == 0:
    print("Can't plot memory data. No data folders found in", rootPath)

  obj = ArithmeticTestPlotter(rootPath, saveDir, dataDirs, pdfName, testNames,\
                              graphHeight=analysisConfig.graphDict["graphHeight"])
  obj.graphAndSaveData()


#given a path, look for all arithmetic-test output files in it's subdirectories.
#if there are directories to not search in, then add that directory
#  as a string to ignoreDirectories param
#Return: TestSpreadCalculator object to grab result dictionaries from
def arithmeticTestSpreads(rootPath, ignoreDirectories=[]):
  
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


def memoryTestSpreads(rootPath, ignoreDirectories=[]):
  testSpreads = TestSpreadCalculator(analysisConfig.memoryOutputFiles, rootPath, ignoreDirectories)
  testSpreads.findRuntimeSpreads()
  testSpreads.findPowerSpreads()
  testSpreads.findEnergySpreadsOfResults()
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
  # control = 'outputAddFP32_1.csv'
  # test = 'outputAddFP32_2.csv'

  table = "\\begin{tabular}{|l|p{0.8in}|p{0.8in}|} \hline\n"
  
  table+= "Measurement  & "+col1Name+" & "+col2Name+"\\\ \hline\n"
  
  table+= "Control Kernel Power & " + col1["controlPow"][0] + "$\pm$" + col1["controlPow"][1] + "\% W & " + col2["controlPow"][0] + "$\pm$" + col2["controlPow"][1] + "\% W\\\ \hline\n"
  table+= "Control Kernel Time & "+ col1["controlTime"][0] + "$\pm$"+ col1["controlTime"][1]+"\% s & "+ col2["controlTime"][0] + "$\pm$"+ col2["controlTime"][1]+"\% s\\\ \hline\n"
  table+= "Control Kernel Energy & "+ col1["controlEnergy"][0] + "$\pm$"+ col1["controlEnergy"][1]+"\% J & "+ col2["controlEnergy"][0] + "$\pm$"+ col2["controlEnergy"][1]+"\% J\\\ \hline\n"
  
  table+= "Test Kernel Power & " + col1["testPow"][0] + "$\pm$" + col1["testPow"][1] + "\% W & " + col2["testPow"][0] + "$\pm$" + col2["testPow"][1] + "\% W\\\ \hline\n"
  table+= "Test Kernel Time & "+ col1["testTime"][0] + "$\pm$"+ col1["testTime"][1]+"\% s & "+ col2["testTime"][0] + "$\pm$"+ col2["testTime"][1]+"\% s\\\ \hline\n"
  table+= "Test Kernel Energy & "+ col1["testEnergy"][0] + "$\pm$"+ col1["testEnergy"][1]+"\% J & "+ col2["testEnergy"][0] + "$\pm$"+ col2["testEnergy"][1]+"\% J\\\ \hline\n"
  table+= "Base Power & "+ col1["basePow"][0] + "$\pm$"+ col1["basePow"][1]+"\% J & "+ col2["basePow"][0] + "$\pm$"+ col2["basePow"][1]+"\% J\\\ \hline\n"

  table+= "Test Kernel Marginal Energy & " + col1["marginalEnergy"][0] + "$\pm$" + col1["marginalEnergy"][1] + "\% J & " + col2["marginalEnergy"][0] + "$\pm$" + col2["marginalEnergy"][1] + "\% J\\\ \hline\n"

  table+= "Marginal Operations Per Kernel & " + col1["marginalOps"] + "& " + col2["marginalOps"] + " \\\ \hline\n"
  table+= "Energy Per Operation & " + col1["energyPerOp"][0] + "$\pm$" + col1["energyPerOp"][1] + "\% pJ & " + col2["energyPerOp"][0] + "$\pm$" + col2["energyPerOp"][1] + "\% pJ\\\ \hline\n"
  
  table+= "\end{tabular}\n"

  return table

def makeAbreviatedTable(results, deviceNames):
  table = {} #key = name of test, value = string representing row of that test in file

  #setup result saving to table
  for i, name in enumerate(analysisConfig.arithTestNames): 
    table[name] = name.replace("FP32", "SP").replace("FP64", "DP") #this makes names more readable
  table["beg"] = "\\begin{tabular}{|l|"
  table["header"] = " "

  #put res into table
  for i, name in enumerate(deviceNames):
    deviceRes = results[name]
    table["header"] += " & " + name
    table["beg"] += "p{0.8in}|" #make header wider
    for test, col in deviceRes.items():
      table[test] += " & " + col["energyPerOp"][0] + "$\pm$" + col["energyPerOp"][1] + "\% pJ"

  #finishup
  res = ""
  table["beg"] += "} \hline\n"
  table["header"] += "\\\ \hline\n"
  res += table["beg"]
  res += table["header"]
  for i, test in enumerate(analysisConfig.arithTestNames):
    table[test] += "\\\ \hline\n"
    res += table[test]
  table["end"] = "\end{tabular}\n"
  res += table["end"]

  return res


def analyzeData():
  rootPath = analysisConfig.pathDict["baseDir"]
  saveDir = rootPath + analysisConfig.pathDict["saveDir"]
  dataDirs = glob.glob(rootPath + analysisConfig.pathDict["dataDirs"])

  pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True) 
  print("Analyzing data from directory: '" + rootPath + "'")
  print("   Saving analysis output in: '" + saveDir + "'")
  print("   Looking in directories named like: '" + analysisConfig.pathDict["dataDirs"] + "'\n")

  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders in", rootPath, "found" )
    exit(1)

  # testSpreadsObj2 = arithmeticTestSpreads("testRuns/p6000_eigth_set/")
  # testSpreadsObj2 = arithmeticTestSpreads("testRuns/k20_eigth_set/")

  # kernelBPAnalysis = BasePowForKernels(rootPath, dataDirs, saveDir, [1,2,3,4], analysisConfig.basePow2ResultFiles, 2)
  # kernelBPAnalysis.calcBasePows()
  # basePowResults = kernelBPAnalysis.getResults()
  # print(basePowResults)
  # # quit(0)

  '''
  arithResults = {} #key = device name, value = dict of arith results
  arithResults["k20"] = {} #key = test name, value = results as a dict
  arithResults["p6000"] = {}
  arithTestSpreadsObj = arithmeticTestSpreads(rootPath)
  for name, (control, test) in analysisConfig.arithTestNamesToFiles.items():
  # for control, test in analysisConfig.arithOutputPairs:
    # col = makeTableColEntry(basePowResults[name][0][2:], arithTestSpreadsObj, control, test)
    try:
      col = makeTableColEntry((80.0,0), arithTestSpreadsObj, control, test)
      arithResults["k20"][name] = col
      # col2 = makeTableColEntry((95.0,0), testSpreadsObj2, control, test)
      # arithResults["p6000"][name] = col2
      # cols.append(col2)
      # names.append("p6000")
      # print("$"+name+"$\\\ \n"+makeTableFromCols(col, col, "K20", "K20"))
    except IndexError as err:
      print("IndexError: failed creating table for: '"+str(err)+"'")
    except KeyError as err:
      print("KeyError: failed creating table for: '"+str(err)+"'")
  a = arithResults["k20"]
  arithResults["p6000"] = a
  print(makeAbreviatedTable(arithResults, ["k20", "p6000"]))
  '''


  # memoryTestSpreadsObj = memoryTestSpreads(rootPath)
  # for name, (control, test) in analysisConfig.memoryTestNamesToFiles.items():
  #   try:
  #     col = makeTableColEntry((35.0,0), memoryTestSpreadsObj, control, test)
  #     col2 = makeTableColEntry((40.0,0), memoryTestSpreadsObj, control, test)
  #     print("$"+name+"$\\\ \n"+makeTableFromCols(col, col2, "K20_35", "K20_40"))
  #   except IndexError as err:
  #     print("IndexError: failed creating table for: '"+str(err)+"'")
  #   except KeyError as err:
  #     print("KeyError: failed creating table for: '"+str(err)+"'")


  calculateBasePower(rootPath, saveDir, dataDirs) #this hasn't been updated for newer file names with different kernels
  # graphBasePower(rootPath, saveDir, dataDirs)
  # graphArithmetic(rootPath, saveDir, dataDirs)
  # graphMemory(rootPath, saveDir, dataDirs)


if __name__ == "__main__":
  # analyzeData()


  # print("Calculating base power from approach 1")
  # obj = BasePowForKernels("testing/bpTests", [1,2], analysisConfig.basePow1ResultFiles, 1)
  # obj = BasePowForKernels("testing/bpTests/", [1,2], {"addFP32":"basePow1_addFloat.csv"}, 1)

  # print("Calculating base power from approach 2")
  # obj = BasePowForKernels("testing/bpTests/", [1,2], analysisConfig.basePow2ResultFiles, 2)
  # obj = BasePowForKernels("testing/bpTests/", [1,2], {"addFP32":"basePow2_addFloat.csv"}, 2)
  # obj.calcBasePows()
  # def __init__(self, rootPath, dataDirs, storagePath, testIDs, dataNameDict, basePowMethod):

  print("Calculating base power from approach 1")
  obj = BasePowForKernels("testRuns/p6000_eigth_set/", ["run1/"], "analysis/", [1,2], analysisConfig.basePow1ResultFiles, 1)
  # obj = BasePowForKernels("testRuns/p6000_eigth_set/", "run1/", "analysis/output.txt",[1,2], {"addFP32":"basePow1_addFloat.csv"}, 1)

  # print("Calculating base power from approach 2")
  # obj = BasePowForKernels("testRuns/p6000_eigth_set/", "run1/", "analysis/output.txt",[1,2], analysisConfig.basePow2ResultFiles, 2)
  # obj = BasePowForKernels("testRuns/p6000_eigth_set/", "run1/", "analysis/output.txt",[1,2], {"addFP32":"basePow2_addFloat.csv"}, 2)
  obj.calcBasePows()






