


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