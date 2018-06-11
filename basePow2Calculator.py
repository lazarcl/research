import glob
import pandas
import itertools


class BasePow2Calculator(object):
	"""
	Calculate the base power from data collected. Data was collected using the
	second approach. Inputs is the path to the folder where the data to process
	is stored, and the runs that should be examined.				
	"""
	def __init__(self, pathToData, runIDs):
		super(BasePow2Calculator, self).__init__()
		#path to folder where data to examine is held
		self.pathToData = pathToData
		if pathToData[-1] != "/":
			pathToData+="/"

		#list of ints representing the run numbers
		self.runIDs = runIDs

		#dict that hold the data in arrays. Key=runID, value=power data array
		self.data = {}

		#dict to hold avg power value for each run. key=runID, value=avg power value
		self.runAverages = {}

		#dict to hold the elapsed time for each run. key=runID, value=total time in sec
		self.runTimes = {}

		#dict to hold total energy(KE) for each run. key=runID, value=energy in W
		self.runEnergy = {}

		#when calculating avgs, how many samples to skip at beg and end of data
		self.rampUpSize = 50

		#array where results are stored as tuples: (runJ, runK, BP)
		self.results = []

	#find relevant file names and load into data dict
	def loadData(self):
		for runID in self.runIDs:
			fileName = glob.glob(self.pathToData+"*"+str(runID)+".csv")

			if len(fileName) == 0:
				print("run '"+str(runID)+"' not found in path '"+self.pathToData)
				continue

			colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
			fileData = pandas.read_csv(fileName[0], names=colnames, encoding='utf-8')
			power = fileData.power.tolist()[1:]
			power = [float(power[i]) for i in range(len(power))]

			self.runTimes[runID] = float(fileData.totalT.tolist()[1]) / 1000
			self.data[runID] = power


  #find the average data
	def findAverages(self):
		for runID, runData in self.data.items():
			total = 0
			for i in runData[self.rampUpSize:-self.rampUpSize]: #ignore ramp up/down
				total += i
			self.runAverages[runID] = total/len(runData[self.rampUpSize:-self.rampUpSize]) 

	# calculate all the energy used during the test's kernel run
	def getTotalEnergy(self):
		self.loadData()
		self.findAverages()
		for runID in self.runIDs:
			self.runEnergy[runID] = self.runAverages[runID]*self.runTimes[runID]

	#find base powers between each run and store as a tuple: (run1ID, run2ID, BP)
	def findBasePowers(self):
		self.getTotalEnergy()

		for (j, k) in list(itertools.combinations(self.runIDs, 2)):
			numerator = k*self.runEnergy[j] - j*self.runEnergy[k]
			denom = k*self.runTimes[j] - j*self.runTimes[k]
			self.results.append( (j, k, abs(numerator/denom)) )


	def printBasePowers(self):
		print([round(i[2],2) for i in self.results])

	def getAvgs(self):
		return self.runAverages

	def getEnergy(self):
		return self.runEnergy

	def getTimes(self):
		return self.runTimes


# obj = BasePow2Calculator("data/basePow2/", [3,4,5])
# obj.findBasePowers()
# obj.printBasePowers()







