


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
arithColumnNames = ['power', 'temp', 'time', 'totalT', 'totalSamples', 'numOfOps', 'numOfThreads']


