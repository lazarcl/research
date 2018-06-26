


'''
Specify directories to find or store data to.
saveDir and dataDirs are assumed to be inside of baseDir.
All paths should end with a forward slash.
dataDirs - should be a general, glob, path to find desired data paths inside of
'''
pathDict = dict(
	baseDir = "testRuns/k20_second_set/",
	saveDir = "analysis/",
	dataDirs = "run*/"
)

'''
Settings for output graphs.

'''
graphDict = dict(
  graphHeight = 80 #Max range of the y-axis
)



#---------Expected data file names-----------
basePowerAnalysisFilename = "basePowerResults.txt"
arithGenericGraphPdfName = "arithmeticGraphs_"

basePower1GenericName = "outputBlksPerSM_"
basePower2GenericName = "outputBlockScalar_"

#csv files that contain the basepow1 run results for given kernel
basePow1ResultFiles = dict(
  AddFP32 =   "basePow1_addFloat.csv", 
  AddFP64 =   "basePow1_addDouble.csv",
  AddInt32 =  "basePow1_addInt.csv",
  FMAFP32 =   "basePow1_fmaFloat.csv",
  FMAFP64 =   "basePow1_fmaDouble.csv", 
  MultFP32 =  "basePow1_multFloat.csv",
  MultFP64 =  "basePow1_multDouble.csv",
  MultInt32 = "basePow1_multInt.csv" 
)


arithTestNamesToFiles = dict(
  AddFP32 =   ('outputAddFP32_1.csv', 'outputAddFP32_2.csv'), 
  AddFP64 =   ('outputAddFP64_1.csv', 'outputAddFP64_2.csv'),
  AddInt32 =  ('outputAddInt32_1.csv', 'outputAddInt32_2.csv'),
  FMAFP32 =   ('outputFMAFP32_1.csv', 'outputFMAFP32_2.csv'),
  FMAFP64 =   ('outputFMAFP64_1.csv', 'outputFMAFP64_2.csv'), 
  MultFP32 =  ('outputMultFP32_1.csv', 'outputMultFP32_2.csv'),
  MultFP64 =  ('outputMultFP64_1.csv', 'outputMultFP64_2.csv'),
  MultInt32 = ('outputMultInt32_1.csv', 'outputMultInt32_2.csv') 
)
arithTestNames = [name for name, files in arithTestNamesToFiles.items()]
arithOutputPairs = [files for name, files in arithTestNamesToFiles.items()]
arithOutputFiles = [x for t in arithOutputPairs for x in t]

arithColumnNames = ['power', 'temp', 'time', 'totalT', 'totalSamples', 'numOfOps', 'numOfThreads']
basePowColumnNames = ['runID', 'avgPower', 'totalT']


