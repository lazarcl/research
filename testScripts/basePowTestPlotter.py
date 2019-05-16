import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages
import glob
import csv
import os
import testScripts.dataLoader as dataLoader


'''
helpful styling guide:

http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
'''

class BasePowTestPlotter:


  def __init__(self, saveDir, dataDirs, graphHeight=160):
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
    self.MAX_Y = graphHeight
    self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "tab:grey"]


  #name of test to make graph for
  #  ex: FMAFP32, FMAFP64, AddInt32, MultFP32...
  def makeFigure(self, testPaths, supTitle, subTitle):
    powerLists = []
    for path in testPaths:
      #append (powerData, timeData)
      powerLists.append(list(dataLoader.getPowerAndTimeFromFile(path)) + [path[-5]] )

    fig = pylab.figure()
    ax = pylab.subplot(111)    

    # ax.spines["top"].set_visible(False)    
    # ax.spines["right"].set_visible(False)    

    powerLists.sort(key=lambda x: int(x[2]))
    for i in range(len(powerLists)):
      # pylab.plot(powerLists[i][1], powerLists[i][0], self.colors[i], label="workload of "+str(i+1)+"x", lw=self.LINE_WIDTH)
      pylab.plot(powerLists[i][1], powerLists[i][0], self.colors[i], label="workload of "+powerLists[i][2]+"x", lw=self.LINE_WIDTH)

    pylab.xlabel('time (ms)')
    pylab.ylabel('power (W)')
    # if subTitle is not "":
    #   pylab.suptitle(supTitle)
    #   pylab.title(subTitle, fontsize=6)
    # else:
    #   pylab.title(supTitle)

    # pylab.legend(loc="lower right")
    pylab.ylim(0, self.MAX_Y)
    # pylab.ylim(0, 65)
    # fig.set_size_inches(7.0, 3.8)


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
      pp.savefig(fig, dpi=self.SAVE_DPI, bbox_inches='tight')
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
    # supTitle = "Base Power 1st Approach Run"
    # subTitle = "Changing number of concurrent blocks per SM"
    supTitle = "Identical Workloads with Changing Block Concurrency"

    self.makeBasePowGraphsGeneral(supTitle, '', generalFileName, saveAs)
    # self.makeBasePowGraphsGeneral(supTitle, subTitle, generalFileName, saveAs)


  def makeBasePow2Graphs(self):
    generalFileName = "outputBlockScalar_*.csv"
    saveAs = "resultsGraphBP2.pdf"
    # supTitle = "Base Power 2nd Approach Run"
    # subTitle = "Linearly changing number of blocks per run"
    supTitle= "Linearly Increasing Workloads"

    self.makeBasePowGraphsGeneral(supTitle, '', generalFileName, saveAs)
    # self.makeBasePowGraphsGeneral(supTitle, subTitle, generalFileName, saveAs)




if __name__ == "__main__":

  basePath = "testRuns/p6000_second_set/"
  saveDir = basePath + "analysis/"

  dataDirs = glob.glob(basePath + "run1/")
  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders in", basePath, "found" )

  obj = BasePowTestPlotter(saveDir, dataDirs)
  # obj.makeBasePow2Graphs()
  obj.makeBasePow1Graphs()
    
  
















