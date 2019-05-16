

'''
Specify directories to find or store data to.
saveDir and dataDirs are assumed to be inside of baseDir.
All paths should end with a forward slash.
dataDirs - should be a general, glob, path to find desired data paths inside of
'''
pathDict = dict(
	# baseDir = "testRuns/p6000_second_set/",
  # baseDir = "testRuns/k20_second_set_old/",
	# baseDir = "testRuns/_mult32/",
	baseDir = "testRuns/k20_third_set/", #pretty good data?
  # baseDir = "testRuns/k20_fifth_set/",
  # baseDir = "testRuns/k20_seventh_set/",
  # baseDir = "testRuns/p6000_seventh_set/",
  # baseDir = "testRuns/memTests_first/",
	# baseDir = "testRuns/k20_fourth_set_multInt32_basePow/",
	# baseDir = "testRuns/k20_fourth_set_multFP32_basePow/",
	# baseDir = "testRuns/testSet/",
  # baseDir = "testRuns/p6000_eigth_set/",
  # baseDir = "testRuns/k20_eigth_set/",
  # baseDir = "testRuns/simbench/p6000_first_set/",
	saveDir = "analysis/",
	dataDirs = "run*/"
)

'''
Settings for output graphs.

'''
graphDict = dict(
  # graphHeight = 180 #Max range of the y-axis
  graphHeight = 80 #Max range of the y-axis
)




#---------Expected data file names-----------
basePowerAnalysisFilename = "basePowerResults.txt"
arithGenericGraphPdfName = "arithmeticGraphs_"
memoryGenericGraphPdfName = "memoryGraphs_"

basePower1GenericName = "outputBlksPerSM_"
basePower2GenericName = "outputBlockScalar_"

#csv files that contain the basepow1 run results for given kernel
basePow1ResultFiles = dict(
  AddFP32 =   "basePow1_addFloat.csv", 
  AddFP64 =   "basePow1_addDouble.csv",
  AddInt32 =  "basePow1_addInt.csv",
  MultFP32 =  "basePow1_multFloat.csv",
  MultFP64 =  "basePow1_multDouble.csv",
  MultInt32 = "basePow1_multInt.csv", 
  FMAFP32 =   "basePow1_fmaFloat.csv",
  FMAFP64 =   "basePow1_fmaDouble.csv"
)

#csv files that contain the basepow1 run results for given kernel
basePow2ResultFiles = dict(
  AddFP32 =   "basePow2_addFloat.csv", 
  AddFP64 =   "basePow2_addDouble.csv",
  AddInt32 =  "basePow2_addInt.csv",
  MultFP32 =  "basePow2_multFloat.csv",
  MultFP64 =  "basePow2_multDouble.csv",
  MultInt32 = "basePow2_multInt.csv", 
  FMAFP32 =   "basePow2_fmaFloat.csv",
  FMAFP64 =   "basePow2_fmaDouble.csv"
)

arithTestNamesToFiles = dict(
  AddFP32 =   ('outputAddFP32_1.csv', 'outputAddFP32_2.csv'), 
  AddFP64 =   ('outputAddFP64_1.csv', 'outputAddFP64_2.csv'),
  AddInt32 =  ('outputAddInt32_1.csv', 'outputAddInt32_2.csv'),
  MultFP32 =  ('outputMultFP32_1.csv', 'outputMultFP32_2.csv'),
  MultFP64 =  ('outputMultFP64_1.csv', 'outputMultFP64_2.csv'),
  MultInt32 = ('outputMultInt32_1.csv', 'outputMultInt32_2.csv'),
  FMAFP32 =   ('outputFMAFP32_1.csv', 'outputFMAFP32_2.csv'),
  FMAFP64 =   ('outputFMAFP64_1.csv', 'outputFMAFP64_2.csv')
)

arithTestNames = [name for name, files in arithTestNamesToFiles.items()]
arithOutputPairs = [files for name, files in arithTestNamesToFiles.items()]
arithOutputFiles = [x for t in arithOutputPairs for x in t]

arithColumnNames = ['power', 'temp', 'time', 'totalT', 'totalSamples', 'numOfOps', 'numOfThreads']
basePowColumnNames = ['runID', 'avgPower', 'totalT']

memoryTestNamesToFiles = dict(
  L1CacheReadTest = ('outputL1ReadTest_1.csv','outputL1ReadTest_2.csv'),
  L2CacheReadTest = ('outputL2ReadTest_1.csv','outputL2ReadTest_2.csv'),
  GlobalReadTest = ('outputGlobalReadTest_1.csv','outputGlobalReadTest_2.csv'),
  SharedReadTest = ('outputSharedReadTest_1.csv','outputSharedReadTest_2.csv')
)

memoryTestNames = [name for name, files in memoryTestNamesToFiles.items()]
memoryOutputPairs = [files for name, files in memoryTestNamesToFiles.items()]
memoryOutputFiles = [x for t in memoryOutputPairs for x in t]



