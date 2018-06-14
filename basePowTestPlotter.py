import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages
import glob

'''
helpful styling guide:

http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
'''

##constants
SAVE_DPI = 1000
LINE_WIDTH = 0.4
MAX_Y = 70

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "tab:grey"]



#given filepath, return list of power data as FPs
def getPowerFromFile(filePath):
  colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
  data = pandas.read_csv(filePath, names=colnames, encoding='utf-8')

  power = data.power.tolist()[1:]
  power = [float(power[i]) for i in range(len(power))]
  return power

#name of test to make graph for
#  ex: FMAFP32, FMAFP64, AddInt32, MultFP32...
def makeFigure(testPathTups, supTitle, subTitle):
  powerLists = []
  for i in range(len(testPathTups)):
    powerLists.append(getPowerFromFile(testPathTups[i][0]))

  f = pylab.figure()
  ax = pylab.subplot(111)    
  ax.spines["top"].set_visible(False)    
  ax.spines["right"].set_visible(False)    
  
  for i in range(len(powerLists)):
    pylab.plot(range(len(powerLists[i])), powerLists[i], colors[i], label="workload of "+str(i+1)+"x", lw=LINE_WIDTH)

  pylab.xlabel('time(ms)')
  pylab.ylabel('power(W)')
  pylab.suptitle(supTitle)
  pylab.title(subTitle, fontsize=8)

  pylab.legend(loc="lower right")
  pylab.ylim(0, MAX_Y)

  #save to seperate file
  # f.savefig("data/"+ testStr + ".pdf", bbox_inches='tight')

  return f

#input: list of test strings to make graphs for
#  ex: ["FMAFP32", "AddInt32"]
#desired graphs title and subtitle
#output: list of plots. One plot per input element
def getListOfPlots(listOfTests, supTitle, subTitle):
  plots = []
  for test in listOfTests:
    plots.append(makeFigure(test, supTitle, subTitle))
  return plots

#given a file name to save to, and a list of figures, save to one pdf
def saveFigureList(figs, filePath):
  pp = PdfPages(filePath)
  for fig in figs:
    pp.savefig(fig, dpi=SAVE_DPI)
  pp.close()


def createTestPathTuples(paths, folderPath):
  pathTups = []
  for i in range(len(paths)):
    name = paths[i].replace(folderPath+"output", "").replace(".csv", "")
    pathTups.append( (paths[i], name) )
  return pathTups


def makeBasePow2Graph(folderPath):
  if folderPath[-1] != "/":
    folderPath+="/"

  supTitle = "Base Power 2nd Approach Run"
  subTitle = "Linearly changing number of blocks per run"
  paths = glob.glob(folderPath+"outputBlockScalar_*.csv")

  pathTuples = createTestPathTuples(paths, folderPath)
  fig = makeFigure(pathTuples, supTitle, subTitle)
  fig.savefig(folderPath+"resultsGraphBP2.pdf", dpi=SAVE_DPI)

def makeBasePow1Graph(folderPath):
  if folderPath[-1] != "/":
    folderPath+="/"

  supTitle = "Base Power 1st Approach Run"
  subTitle = "Changing number of concurrent blocks per SM"
  paths = glob.glob(folderPath+"outputBlksPerSM_*.csv")

  pathTuples = createTestPathTuples(paths, folderPath)
  fig = makeFigure(pathTuples, supTitle, subTitle)
  fig.savefig(folderPath+"resultsGraphBP1.pdf", dpi=SAVE_DPI)



if __name__ == "__main__":
  for folderPath in glob.glob("testRuns/run*/"):
    makeBasePow2Graph(folderPath)
  for folderPath in glob.glob("testRuns/run*/"):
    makeBasePow1Graph(folderPath)












