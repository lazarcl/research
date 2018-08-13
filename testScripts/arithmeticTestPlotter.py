import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages
import glob
import testScripts.dataLoader as dataLoader


'''
helpful styling guide:

http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
'''

class ArithmeticTestPlotter:

  #rootPath: the path to the data and analysis folders
  #savePath: the path to save the results pdf to
  #dataDirs: list of paths to the directories where source data is stored
  #testNames: list of the test names to make graphs for.
    #Names must be the '*' part of 'output*_1.csv' and 'output*_2.csv' for data matching
  def __init__(self, rootPath, savePath, dataDirs, pdfName, testNames, graphHeight=160):
    ##constants
    self.SAVE_DPI = 1000
    self.LINE_WIDTH = 0.4
    self.MAX_Y = graphHeight

    self.rootPath = rootPath
    self.savePath = savePath
    self.dataDirs = dataDirs
    self.pdfName = pdfName
    self.testNames = testNames

    if self.rootPath[-1] != "/":
      self.rootPath+="/"
    if self.savePath[-1] != "/":
      self.savePath+="/"
    for i in range(len(self.dataDirs)):
      if self.dataDirs[i][-1] != "/":
        self.dataDirs[i]+="/"


  #Load the data for given files and return graphed figure
  #  ex: FMAFP32, FMAFP64, AddInt32, MultFP32...
  def makeFigure(self, file1, file2, testName):
    # power1, time1 = self.getPowerAndTimeFromFile(file1)
    # power2, time2 = self.getPowerAndTimeFromFile(file2)
    power1, time1 = dataLoader.getPowerAndTimeFromFile(file1)
    power2, time2 = dataLoader.getPowerAndTimeFromFile(file2)

    f = pylab.figure()
    ax = pylab.subplot(111)    
    ax.spines["top"].set_visible(False)    
    # ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    # ax.spines["left"].set_visible(False)    

    pylab.plot(time1, power1, "-b", label="Control Kernel", lw=self.LINE_WIDTH)
    pylab.plot(time2, power2, "-r", label="Test Kernel", lw=self.LINE_WIDTH)


    pylab.xlabel('time(ms)')
    pylab.ylabel('power(W)')
    pylab.title(testName + " Kernel Runs")

    pylab.legend(loc="lower right")
    pylab.ylim(0, self.MAX_Y)

    #save to seperate file
    # f.savefig("data/"+ testStr + ".pdf", bbox_inches='tight')
    return f

  #for given directory path, create return figures for all tests inside
  def getListOfPlots(self, dataDir):
    plotList = []
    for test in self.testNames:
      try:
        file1 = dataDir + "output" + test.replace('Cache', '') + "Test_1.csv"
        file2 = dataDir + "output" + test.replace('Cache','') + "Test_2.csv"
        fig = self.makeFigure(file1, file2, test)
        plotList.append(fig)
      except FileNotFoundError as err:
        print("Error Plotting Arithmetic Tests: \n  " + str(err).replace("File b", '')+". Continuing..." )
    return plotList


  #for each directory, make graphs and save them to savePath.
  #Figures are deleted after saving
  def graphAndSaveData(self):
    for i in range(len(self.dataDirs)):
      plotList = self.getListOfPlots(self.dataDirs[i])
      self.saveFigures(plotList, i+1)
      for fig in plotList:
        pylab.close(fig)


  #given a file name to save to, and a list of figures, save to one pdf
  def saveFigures(self, figs, pdfNameNum):
    pdf = PdfPages(self.savePath + self.pdfName + str(pdfNameNum) + ".pdf")
    for fig in figs:
      pdf.savefig(fig, dpi=self.SAVE_DPI)
    pdf.close()

  def saveFigureLists(self, listsOfFigs):
    for i in range(len(listsOfFigs)):

      pdf = PdfPages(self.savePath + self.pdfName + str(i+1) + ".pdf")
      for fig in listsOfFigs[i]:
        pdf.savefig(fig, dpi=self.SAVE_DPI)
      pdf.close()



if __name__ == "__main__":
  testNames = ["AddFP32", "AddFP64", "AddInt32", "FMAFP32", "FMAFP64", "MultFP32", "MultFP64", "MultInt32"]
  
  rootPath = "testRuns/k20_second_set/"
  savePath = rootPath + "analysis/"
  dataDirs = glob.glob(rootPath + "run*/")

  if len(dataDirs) == 0:
    print("Can't plot arithmetic data. No data folders found in", rootPath)

  pdfName = "arithmeticGraphs_"
  obj = ArithmeticTestPlotter(rootPath, savePath, dataDirs, pdfName, testNames)
  obj.graphAndSaveData()

  # obj = ArithmeticTestPlotter(rootPath, savePath, dataDirs, pdfName, testNames)
  # for i in range(len(dataDirs)):
  #   listsOfPlots = obj.getListsOfPlots()
  #   saveAs = savePath + "arithmeticGraphs_" + str(i) + ".pdf"
  #   obj.saveFigureLists(listsOfPlots)


  # for folderPath in dataDirs:
  # for i in range(len(dataDirs)):
  #   figs = getListOfPlots(dataDirs[i])
  #   if len(figs) == 0:
  #     print("Arithmeic test graph not made. No figures plotted")
  #     exit(1)
  #   else:
  #     print("saving "+str(len(figs))+" arithmetic test graphs")
  #   saveAs = "arithmeticGraphs_" + str(i) + ".pdf"
  #   saveFigures(figs, saveAs)












