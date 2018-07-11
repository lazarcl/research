

import glob
import itertools
import statistics
import testScripts.analysisConfig as analysisConfig
import testScripts.dataLoader as dataLoader
from testScripts.mathHelpers import *


class BasePowForKernels(object):
  """
  Calculate the base power from data collected. Find variance between different runs
  of the same tests. Inputs is the paths to the folders where the data to process
  is stored, and the runs that should be examined.
  All result files should end with '<test_number>.csv'
  """
  def __init__(self, pathToData, testIDs, dataNameDict, basePowMethod):

    #dictionary: key = kernelName, value = fileName for kernel's basePow data
    self.dataNameDict = dataNameDict
 
    #path to folder where data to examine is held
    self.pathToData = pathToData
    if self.pathToData[-1] != "/":
      self.pathToData+="/"

    #list of ints representing the run numbers to inspect
    self.testIDs = testIDs

    #when calculating avgs, how many samples to skip at beg and end of data
    self.rampUpSize = 50

    #which base power calculation to use. 1 for basePow1, 2 for basePow2
    if basePowMethod != 1 and basePowMethod != 2:
      print("Error: while calculating base powers, passed invalid 'basePowMethod' value")
      quit(1)

    self.basePowMethod = basePowMethod

    #basePower results are stored here. Lookup list of results with KernelName
    self.results = {}


  #calc the mean and var in the list provided.
  #if the list contains only one item, then variance is 0
  def calcMeanAndVar(self, valueList):
    mean = statistics.mean(valueList)

    if len(valueList) < 2:
      return mean, 0.0
    else:
      var = statistics.variance(valueList)
      return mean, var

  def calcBasePowForGiven_bp1(self, energys, times):
    results = []
    for (j, k) in list(itertools.combinations(self.testIDs, 2)):
      numer = subIndVar(energys[j], energys[k])
      denom = subIndVar(times[j], times[k])
      mean, var = divIndVar(numer, denom)
      results.append( (j, k, abs(mean), var) )
    return results

  def calcBasePowForGiven_bp2(self, energys, times):
    results = []
    for (j, k) in list(itertools.combinations(self.testIDs, 2)):
      numer = subIndVar(multIndVarAndConst(energys[j], k), multIndVarAndConst(energys[k], j))
      denom = subIndVar(multIndVarAndConst(times[j], k), multIndVarAndConst(times[k], j))
      mean, var = divIndVar(numer, denom)
      results.append( (j, k, abs(mean), var) )
    return results


  def getBasePow(self, dataFile):
    runData = dataLoader.readBasePowData(self.pathToData + dataFile)
    testEnergys = {}
    testTimes = {}
    for testID, (runPowers, runTimes) in runData.items():
      power = self.calcMeanAndVar(runPowers)
      time = self.calcMeanAndVar(runTimes)
      energy = multiplyIndVar(power, time)
      testEnergys[testID] = energy
      testTimes[testID] = time

    try:
      if self.basePowMethod == 1:
        return self.calcBasePowForGiven_bp1(testEnergys, testTimes)
      elif self.basePowMethod == 2:
        return self.calcBasePowForGiven_bp2(testEnergys, testTimes)
      else:
        print("Error: while calculating base powers, passed invalid 'basePowMethod' value")

    except KeyError as err:
      print("Error: While calculating base powers: attempted to calculate base pow between data that doesn't exist. Key: '"+str(err)+"'")
      quit(1)
      

  def calcBasePows(self):
    self.results = {}
    with open(self.pathToData+"basePowResults.txt", "w") as f:
      for kernelName, dataFile in self.dataNameDict.items():
        bps = self.getBasePow(dataFile)

        self.results[kernelName] = bps

        self.writeBasePowers(f, kernelName)
        for bp in bps:
          self.writeBasePowers(f, bp)
        self.writeBasePowers(f, "")


  def getResults(self):
    return self.results

  #savePath: path/to/file that results should be saved in
  #results are lists of data
  def writeBasePowers(self, file, text):
    file.write("  "+str(text)+"\n")
    file.write("\n")



if __name__ == "__main__":
  basePath = "testing/"

  print("Calculating base power from approach 1")

  # obj = basePowForKernels(basePath, [1,2,3,4], analysisConfig.basePow1ResultFiles, 1)
  obj = BasePowForKernels(basePath, [1,2,3,4], {"multFloat":"basePow1_multFloat.csv"}, 1)
  obj.calcBasePows()
  print("Results for basePow 1:")





