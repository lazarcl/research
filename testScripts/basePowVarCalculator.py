import glob
import pandas
import itertools
import statistics


class BasePowVarCalculator(object):
  """
  Calculate the base power from data collected. Find variance between different runs
  of the same tests. Inputs is the paths to the folders where the data to process
  is stored, and the runs that should be examined.
  All result files should end with '<test_number>.csv'
  """
  def __init__(self, pathsToData, runIDs, runName):
    super(BasePowVarCalculator, self).__init__()

    #generic name of the data files to load. leave out index
    # ex: 'outputBlksPerSM_' to get 'outputBlksPerSM_1.csv'
    self.runName = runName
 
    #path to folder where data to examine is held
    self.pathsToData = pathsToData
    for i in range(len(self.pathsToData)):
      if self.pathsToData[i][-1] != "/":
        self.pathsToData[i]+="/"

    #list of ints representing the run numbers
    self.runIDs = runIDs

    #when calculating avgs, how many samples to skip at beg and end of data
    self.rampUpSize = 50

    #array where results are stored as tuples: (runJ, runK, BP, variance)
    self.results = []

  #find relevant file names and load into data dict. return (data,time) tuple
  #return power data in 
    #dict that hold the data in arrays. Key=runID, value=power data array
  #return time in 
    #dict to hold the elapsed time for each run. key=runID, value=total time in sec
  def loadDataAtPathIdx(self, folderIdx):
    data = {}
    runTimes = {}
    dataPath = self.pathsToData[folderIdx]
    for runID in self.runIDs:
      fileName = glob.glob(dataPath+self.runName+str(runID)+".csv")

      if len(fileName) == 0:
        print("run '"+str(runID)+"' not found in path '"+dataPath+"'. Ignoring for calculations")
        self.runIDs.remove(runID)
        continue

      try:
        colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
        fileData = pandas.read_csv(fileName[0], names=colnames, encoding='utf-8')
        power = fileData.power.tolist()[1:]
        power = [float(power[i]) for i in range(len(power))]
        runTimes[runID] = float(fileData.totalT.tolist()[1]) / 1000
        data[runID] = power
      except FileNotFoundError as err:
        print(str(err).replace("File b", ''), "ignoring test and continuing...")
        self.runIDs.remove(runID)
        

    return data, runTimes


  #find the average data
  def findAverages(self, dataDict):
    runAverages = {}
    for runID, runData in dataDict.items():
      runAverages[runID] = statistics.mean(runData[self.rampUpSize:-self.rampUpSize]) 
    return runAverages

      
  def getEnergyForAFolder(self, folderIdx):
    dataDict, runTimes = self.loadDataAtPathIdx(folderIdx)
    runAvgs = self.findAverages(dataDict)
    runEnergys = {}
    for runID in self.runIDs:
      runEnergys[runID] = runAvgs[runID]*runTimes[runID]
    return runEnergys, runTimes


  #add results from given folder index's data to the provided dictionaries
  def combineTestResults(self, pathIdx, combEngyDct, combTimeDct):
    for pathIdx in range(len(self.pathsToData)):
      runEnergys, runTimes = self.getEnergyForAFolder(pathIdx)

      for runID, energy in runEnergys.items():
        if runID not in combEngyDct:
          combEngyDct[runID] = []
        combEngyDct[runID].append(energy)

      for runID, runtime in runTimes.items():
        if runID not in combTimeDct:
          combTimeDct[runID] = []
        combTimeDct[runID].append(runtime)

  #calc the mean and var in the list provided.
  #if the list contains only one item, then variance is 0
  def calcMeanAndVar(self, valueList):
    mean = statistics.mean(valueList)

    if len(valueList) < 2:
      return mean, 0.0
    else:
      var = statistics.variance(valueList)
      return mean, var


  def calcGroupedSamples(self):
    combEngyDct = {} #key=runID, value=array of total energy for the runID
    combTimeDct = {} #key=runID, value=array of run-times's for the runID
    for i in range(len(self.pathsToData)):
      self.combineTestResults(i, combEngyDct, combTimeDct)

    #key=runID, value=(mean,variation) of energy from that runID's runs
    energyCombined = {}
    #key=runID, value=(mean,variation) of runtime from that runID's runs 
    timesCombined = {} 
    for runID, energyList in combEngyDct.items():
      mean, var = self.calcMeanAndVar(energyList)
      # var = statistics.variance(energyList)
      # mean = statistics.mean(energyList)
      energyCombined[runID] = (mean, var)
    for runID, timeList in combTimeDct.items():
      mean, var = self.calcMeanAndVar(timeList)
      # var = statistics.variance(timeList)
      # mean = statistics.mean(timeList)
      timesCombined[runID] = (mean, var)

    return energyCombined, timesCombined


  def calcBasePow(self):
    self.runEnergys, self.runTimes = self.calcGroupedSamples()

    for (j, k) in list(itertools.combinations(self.runIDs, 2)):
      numer = k*self.runEnergys[j][0] - j*self.runEnergys[k][0]
      denom = k*self.runTimes[j][0] - j*self.runTimes[k][0]

      numerVar = k*self.runEnergys[j][1] + j*self.runEnergys[k][1]
      denomVar = k*self.runTimes[j][1] + j*self.runTimes[k][1]

      mean = numer / denom
      var = ( numerVar + (( denomVar * (numer**2) )/(denom**2)) ) / (denom**2)
      self.results.append( (j, k, abs(mean), var) )


  def printBasePowers(self):
    print([(a,b,round(c,2), round(d,2)) for a,b,c,d in self.results])

  def getBasePowers(self):
    return [(a,b,round(c,2), round(d,2)) for a,b,c,d in self.results]


#savePath: path/to/file that results should be saved in
#results are lists of data
def writeBasePowers(basePow1Results, basePow2Results, savePath):
  with open(savePath, 'w') as f:

    f.write("Base Power 1 Results:\n")
    for result in basePow1Results:
      f.write("  "+str(result)+"\n")
    f.write("\n")

    f.write("Base Power 2 Results:\n")
    for result in basePow2Results:
      f.write("  "+str(result)+"\n")

  # f.close()


if __name__ == "__main__":
  # folderPaths = ["data/basePow2/", "data/basePow2_1", "data/basePow2_2", "data/basePow2_3", "data/basePow2_4", "data/basePow2_5"]
  # folderPaths = ["data/basePow1/", "data/basePow1_1", "data/basePow1_2", "data/basePow1_3", "data/basePow1_4"]
  basePath = "testRuns/k20_second_set/"
  dataFolderPaths = glob.glob(basePath + "run*/")
  savePath = basePath + "analysis/basePowerResults.txt"
  print("Calculating base power from approach 1")
  obj = BasePowVarCalculator(dataFolderPaths, [3,4,5], "outputBlksPerSM_")
  obj.calcBasePow()
  # print("Results for basePow 1:")
  # obj.printBasePowers()

  print("\nCalculating base power from approach 2")
  obj2 = BasePowVarCalculator(dataFolderPaths, [3,4,5], "outputBlockScalar_")
  obj2.calcBasePow()
  # print("Results for basePow 2:")
  # obj2.printBasePowers()

  print("Writing base power energy results to:", savePath)
  writeBasePowers(obj.getBasePowers(), obj2.getBasePowers(), savePath)






