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
	def __init__(self, pathsToData, runIDs):
		super(BasePowVarCalculator, self).__init__()

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
	#return data in 
		#dict that hold the data in arrays. Key=runID, value=power data array
	#return time in 
		#dict to hold the elapsed time for each run. key=runID, value=total time in sec
	def loadDataAtPathIdx(self, folderIdx):
		data = {}
		runTimes = {}
		dataPath = self.pathsToData[folderIdx]
		for runID in self.runIDs:
			fileName = glob.glob(dataPath+"*"+str(runID)+".csv")

			if len(fileName) == 0:
				print("run '"+str(runID)+"' not found in path '"+dataPath)
				continue

			colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
			fileData = pandas.read_csv(fileName[0], names=colnames, encoding='utf-8')
			power = fileData.power.tolist()[1:]
			power = [float(power[i]) for i in range(len(power))]

			runTimes[runID] = float(fileData.totalT.tolist()[1]) / 1000
			data[runID] = power

		return data, runTimes


  #find the average data
	def findAverages(self, dataDict):
		runAverages = {}
		for runID, runData in dataDict.items():
			runAverages[runID] = statistics.mean(runData[self.rampUpSize:-self.rampUpSize]) 
			# total = 0
			# for i in runData[self.rampUpSize:-self.rampUpSize]: #ignore ramp up/down
			# 	total += i
			# runAverages[runID] = total/len(runData[self.rampUpSize:-self.rampUpSize]) 
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
			var = statistics.variance(energyList)
			mean = statistics.mean(energyList)
			energyCombined[runID] = (mean, var)
		for runID, timeList in combTimeDct.items():
			var = statistics.variance(timeList)
			mean = statistics.mean(timeList)
			timesCombined[runID] = (mean, var)

		return energyCombined, timesCombined


	def calcAppr2Energy(self):
		self.runEnergys, self.runTimes = self.calcGroupedSamples()

		for (j, k) in list(itertools.combinations(self.runIDs, 2)):
			numer = k*self.runEnergys[j][0] - j*self.runEnergys[k][0]
			denom = k*self.runTimes[j][0] - j*self.runTimes[k][0]

			numerVar = k*self.runEnergys[j][1] + j*self.runEnergys[k][1]
			denomVar = k*self.runTimes[j][1] + j*self.runTimes[k][1]

			mean = numer / denom
			var = ( numerVar + (( denomVar * (numer**2) )/(denom**2)) ) / (denom**2)
			self.results.append( (j, k, abs(mean), var) )


	#find base powers between each run and store as a tuple: (run1ID, run2ID, BP)
	# def findBasePowers(self):
	# 	energyDict = self.getTotalEnergy()

	# 	for pathIdx, runEnergy in energyDict.items():
	# 		for (j, k) in list(itertools.combinations(self.runIDs, 2)):
	# 			numerator = k*runEnergy[j] - j*runEnergy[k]
	# 			denom = k*self.runTimes[j] - j*self.runTimes[k]
	# 			self.results.append( (j, k, abs(numerator/denom)) )


	def printBasePowers(self):
		print([(a,b,round(c,2), round(d,2)) for a,b,c,d in self.results])



if __name__ == "__main__":
	folderPaths = ["data/basePow2/", "data/basePow2_1", "data/basePow2_2", "data/basePow2_3", "data/basePow2_4", "data/basePow2_5"]
	obj = BasePowVarCalculator(folderPaths, [3,4,5])
	obj.calcAppr2Energy()
	obj.printBasePowers()







