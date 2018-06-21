import pandas
import testScripts.analysisConfig as analysisConfig

#functions to load data from output files for analysis





def loadFile(filePath, colnames):
  try:
    data = pandas.read_csv(filePath, low_memory=False, names=colnames, encoding='utf-8')
    return data
  except NameError as err:
    print("DataLoader Error. Cant find file: \n  " + str(err).replace("File b", '')+". Continuing..." )
    return None

def getPowersFromPandas(data):
  power = data.power.tolist()[1:]
  power = [float(power[i]) for i in range(len(power))]
  return power

def getTimesFromPandas(data):
  time = data.time.tolist()[1:]
  time = [float(time[i]) for i in range(len(time))]
  return time


#get elapsed time of given run's data in seconds
#elapsed time expeted to be in 4th column 2nd row (starting at 1)
def getRuntimeFromFile(path):
  data = loadFile(path, analysisConfig.arithColumnNames)
  if data is None:
    return None

  return float(data.totalT[1]) / 1000

def getTotalTimeFromFile(filePath):
  data = loadFile(filePath, analysisConfig.arithColumnNames)
  if data is None:
    return None
  return float(data.totalT.tolist()[1])

#return numOfOps, numOfThreads from file. Caller's responsibility to error check
#returns None, None if file not found, returns nan,nan if fields not filled
def getOpAndThreadCountFromFile(filePath):
  data = loadFile(filePath, analysisConfig.arithColumnNames)
  if data is None:
    return None, None

  numOfOps = int(data.numOfOps.tolist()[1])
  numOfThreads = int(data.numOfThreads.tolist()[1])

  return numOfOps, numOfThreads

def getPowsFromFile(filePath):
  data = loadFile(filePath, analysisConfig.arithColumnNames)
  if data is None:
    return None

  return getPowersFromPandas(data)

#given filepath, return list of power data as FPs
def getPowerAndTimeFromFile(filePath):
  data = loadFile(filePath, analysisConfig.arithColumnNames)
  if data is None:
    return None

  return getPowersFromPandas(data), getTimesFromPandas(data)


  # power = data.power.tolist()[1:]
  # power = [float(power[i]) for i in range(len(power))]
  # time = data.time.tolist()[1:]
  # time = [float(time[i]) for i in range(len(time))]
  # return power, time


# getPowerAndTimeFromFile("testRuns/testSet/run1/outputAddFP32_2.csv")



