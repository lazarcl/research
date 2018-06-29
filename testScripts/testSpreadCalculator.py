import glob
import statistics
import pandas
import testScripts.dataLoader as dataLoader
from testScripts.mathHelpers import multiplyIndVar




'''
Given a glob of file paths, find the mean and variance 
of all the runtimes between all files.

'''

class TestSpreadCalculator:

  '''
  filesToCalculate: list of different filenames, not paths, whos
                    data will be found and data collected for
  directoryToSearchIn: directory path to look for files in
                    any matching file will be included in data
  ignoreDirectories: list of directories to not include in data
  '''
  def __init__(self, filesToCalculateFor, directoryToSearchIn, ignoreDirectories=[]):
    self.filesToCalculateFor = filesToCalculateFor
    self.directoryToSearchIn = directoryToSearchIn
    self.ignoreDirectories = ignoreDirectories

    if self.directoryToSearchIn[-1] != "/":
      self.directoryToSearchIn+= "/"
    for i in range(len(ignoreDirectories)):
      if self.directoryToSearchIn[i][-1] != "/":
        self.directoryToSearchIn[i]+= "/"

    #key = 'filename.csv', value = ['all/paths/to/that/filename.csv']
    self.filePathDict = {}
    for filename in filesToCalculateFor:
      matchingPaths = glob.glob(directoryToSearchIn+"*/"+filename)

      #remove paths containing ignoreDirectories
      for badP in self.ignoreDirectories:
        matchingPaths = [p for p in matchingPaths if badP not in p]

      #save paths under filename for later analysis
      self.filePathDict[filename] = matchingPaths


    #key = 'filename.csv', value = (mean, variance) of elapsed time 
    #       in all found paths to 'filename.csv'
    self.runtimeSpreadDict = {}

    #key='filename.csv', value = (mean, variance) of power 
    # in all paths to that filename
    self.powerSpreadDict = {}

    #key='filename.csv', value = (mean,variance) of tests energy consumption
    #    between all paths to that filename
    self.energySpreadDict = {}


  def findRuntimeOfPaths(self, paths):
    times = []
    for path in paths:
      runtime = dataLoader.getRuntimeFromFile(path)
      if runtime is not None:
        times.append(runtime) 
    return times


  #paths are locations of the same fileName.
  #ex:  myFolder/*/fileName.csv
  def findPowersOfPaths(self, paths):
    powers = []
    for path in paths:
      powerData = dataLoader.getPowsFromFile(path)
      if powerData is not None:
        powers+=powerData
        # avg, var = self.calcMeanAndVar(powerData)
        # if avg != 0.0:
        #   powers.append(avg) 
    return powers



  def calcMeanAndVar(self, valueList):
    mean = 0.0
    var = 0.0
    if len(valueList) >= 1:
      mean = statistics.mean(valueList)
    if len(valueList) >= 2:
      var = statistics.variance(valueList)
    return mean, var

  #for all input filenames, find mean,var for each filename's elapsed time
  def findRuntimeSpreads(self):
    for fileName, paths in self.filePathDict.items():
      times = self.findRuntimeOfPaths(paths)
      mean, var = self.calcMeanAndVar(times)
      self.runtimeSpreadDict[fileName] = (mean, var)

  #for all input filenames, find mean,var for each filename's power across different tests
  #average all power datapoints from all similar files into one mean and var
  def findPowerSpreads(self):
    for fileName, paths in self.filePathDict.items():
      powers = self.findPowersOfPaths(paths)
      mean, var = self.calcMeanAndVar(powers)
      self.powerSpreadDict[fileName] = (mean, var)

  #must be called AFTER findPowerSpreads and findRuntimeSpreads in order to 
  # have data to calculate on.
  #Raises a RuntimeError if the neccisary data is not found in the class
  def findEnergySpreadsOfResults(self):
    if self.runtimeSpreadDict == {} or self.powerSpreadDict == {}:
      print("ERROR: runtime or power spread results not found \n  ensure you generate those results before calculating energy spread")
      raise RuntimeError

    for file, time in self.runtimeSpreadDict.items():
      power = self.powerSpreadDict[file]
      # avgEnergy = avgPower*avgTime
      # newVar = avgEnergy * math.sqrt((v1/m1)**2 + (v2/m2)**2)
      if power == (0.0, 0.0) and time == (0.0,0.0):
        print("spread could not be calculated for: '" + file + "'. Check if it exits")
        continue 
      self.energySpreadDict[file] = multiplyIndVar(power, time)

  def getRuntimeSpreadDict(self):
    return self.runtimeSpreadDict

  def getPowerSpreadDict(self):
    return self.powerSpreadDict

  def getEnergySpreadDict(self):
    return self.energySpreadDict

  def printRuntimeSpreads(self):
    string = self.spreadDictToString(self.runtimeSpreadDict)
    print(string)

  def printPowerSpreads(self):
    string = self.spreadDictToString(self.powerSpreadDict)
    print(string)

  def printEnergySpreads(self):
    string = self.spreadDictToString(self.energySpreadDict)
    print(string)

  #format and write the given list of spread-dictionaries (from this class) to 
  # a specified file path
  def writeToFile(self, dictsToPrint, saveFile):
    with open(saveFile, 'w') as w:
      for d in dictsToPrint:
        w.write(self.spreadDictToString(d))

  def spreadDictToString(self, dictIn):
    string = ''
    for a, b in dictIn.items():
      string += a + " ("+str(round(b[0],2)) + ", " + str(b[1]) + ")" + "\n"
    return string




