import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages
import glob
import csv
import os

'''
helpful styling guide:

http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
'''

class BasePowTestPlotter:


  def __init__(self, saveDir, dataDirs):
    self.saveDir = saveDir
    self.dataDirs = dataDirs

    if saveDir[-1] != "/":
      self.saveDir+="/"
    for i in range(len(self.dataDirs)):
      if dataDirs[i][-1] != "/":
        self.dataDirs[i]+="/"

    ##constants
    self.SAVE_DPI = 1000
    self.LINE_WIDTH = 0.4
    self.MAX_Y = 80 #160
    self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "tab:grey"]


  #given filepath, return list of power data as FPs
  def getPowerFromFile(self, filePath):
    colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
    data = pandas.read_csv(filePath, low_memory=False, names=colnames, encoding='utf-8')

    power = data.power.tolist()[1:]
    power = [float(power[i]) for i in range(len(power))]
    return power


  #name of test to make graph for
  #  ex: FMAFP32, FMAFP64, AddInt32, MultFP32...
  def makeFigure(self, testPaths, supTitle, subTitle):
    powerLists = []
    for path in testPaths:
      powerLists.append(self.getPowerFromFile(path))

    fig = pylab.figure()
    ax = pylab.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    
    for i in range(len(powerLists)):
      pylab.plot(range(len(powerLists[i])), powerLists[i], self.colors[i], label="workload of "+str(i+1)+"x", lw=self.LINE_WIDTH)

    pylab.xlabel('time(ms)')
    pylab.ylabel('power(W)')
    pylab.suptitle(supTitle)
    pylab.title(subTitle, fontsize=6)

    pylab.legend(loc="lower right")
    pylab.ylim(0, self.MAX_Y)

    #display path to folder of graph's dataset
    try:
      dataPath = testPaths[0][:testPaths[0].rindex('/')+1]
      pylab.text(0, 5, dataPath, fontsize=4)    
    except IndexError:
      pass

    return fig

  #input: list of test strings to make graphs for
  #  ex: ["FMAFP32", "AddInt32"]
  #desired graphs title and subtitle
  #output: list of plots. One plot per input element
  def getListOfPlots(self, listsOfPaths, supTitle, subTitle):
    plots = []
    for paths in listsOfPaths:
      plots.append(self.makeFigure(paths, supTitle, subTitle))
    return plots

  #given a file name to save to, and a list of figures, save to one pdf
  def savePlotList(self, figs, filePath):
    pp = PdfPages(filePath)
    for fig in figs:
      pp.savefig(fig, dpi=self.SAVE_DPI)
    pp.close()


  def createTestPathTuples(self, paths):
    pathTups = []
    for i in range(len(paths)):
      name = paths[i].replace(paths+"output", "").replace(".csv", "")
      pathTups.append( (paths[i], name) )
    return pathTups


  def makeBasePowGraphsGeneral(self, supTitle, subTitle, generalFileName, saveAs):

    listsOfDataFiles = []
    for dataFolder in self.dataDirs:
      paths = glob.glob(dataFolder+generalFileName)
      listsOfDataFiles.append(paths)

    plots = self.getListOfPlots(listsOfDataFiles, supTitle, subTitle)
    self.savePlotList(plots, self.saveDir+saveAs)
  

  def makeBasePow1Graphs(self):
    generalFileName = "outputBlksPerSM_*.csv"
    saveAs = "resultsGraphBP1.pdf"
    supTitle = "Base Power 1st Approach Run"
    subTitle = "Changing number of concurrent blocks per SM"

    self.makeBasePowGraphsGeneral(supTitle, subTitle, generalFileName, saveAs)


  def makeBasePow2Graphs(self):
    generalFileName = "outputBlockScalar_*.csv"
    saveAs = "resultsGraphBP2.pdf"
    supTitle = "Base Power 2nd Approach Run"
    subTitle = "Linearly changing number of blocks per run"

    self.makeBasePowGraphsGeneral(supTitle, subTitle, generalFileName, saveAs)




if __name__ == "__main__":

  basePath = "testRuns/k20_sharedMemFix/"
  saveDir = basePath + "analysis/"

  dataDirs = glob.glob(basePath + "run*/")
  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders in", basePath, "found" )

  obj = BasePowTestPlotter(saveDir, dataDirs)
  # obj.makeBasePow2Graphs()
  obj.makeBasePow1Graphs()
    
  
















