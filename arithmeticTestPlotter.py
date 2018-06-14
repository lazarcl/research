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
MAX_Y = 160


#given filepath, return list of power data as FPs
def getPowerFromFile(filePath):
  colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
  data = pandas.read_csv(filePath, names=colnames, encoding='utf-8')

  power = data.power.tolist()[1:]
  power = [float(power[i]) for i in range(len(power))]
  return power

#Load the data for given files and return graphed figure
#  ex: FMAFP32, FMAFP64, AddInt32, MultFP32...
def makeFigure(file1, file2, testName):
  power1 = getPowerFromFile(file1)
  power2 = getPowerFromFile(file2)

  f = pylab.figure()
  ax = pylab.subplot(111)    
  ax.spines["top"].set_visible(False)    
  # ax.spines["bottom"].set_visible(False)    
  ax.spines["right"].set_visible(False)    
  # ax.spines["left"].set_visible(False)    

  pylab.plot(range(len(power1)), power1, "-b", label="Control Kernel", lw=LINE_WIDTH)
  pylab.plot(range(len(power2)), power2, "-r", label="Test Kernel", lw=LINE_WIDTH)


  pylab.xlabel('time(ms)')
  pylab.ylabel('power(W)')
  pylab.title(testName + " Kernel Runs")

  pylab.legend(loc="lower right")
  pylab.ylim(0, MAX_Y)

  #save to seperate file
  # f.savefig("data/"+ testStr + ".pdf", bbox_inches='tight')
  return f


#input: list of test strings to make graphs for
#  ex: ["FMAFP32", "AddInt32"]
#output: list of plots. One plot per input element
def getListOfPlots(folderPath, listOfTests):
  outFigs = []

  for test in listOfTests:
    try:
      file1 = folderPath + "output" + test + "_1.csv"
      file2 = folderPath + "output" + test + "_2.csv"
      fig = makeFigure(file1, file2, test)
      outFigs.append(fig)
    except FileNotFoundError as err:
      print("Error Plotting Arithmetic Tests: \n  " + str(err).replace("File b", '')+". Continuing..." )
  
  return outFigs

#given a file name to save to, and a list of figures, save to one pdf
def saveFigureList(figs, filePath):
  pp = PdfPages(filePath)
  for fig in figs:
    pp.savefig(fig, dpi=SAVE_DPI)
  pp.close()


if __name__ == "__main__":
  testNames = ["AddFP32", "AddFP64", "AddInt32", "FMAFP32", "FMAFP64", "MultFP32", "MultFP64", "MultInt32"]
  folderPaths = glob.glob("testRuns/run*/")

  if len(folderPaths) == 0:
    print("Can't plot arithmetic data. No folders in './testRuns/'")
  else:
    print("plotting arithmetic graphs for folders:")
    for f in folderPaths:
      print("  "+f)

  for folderPath in folderPaths:
    figs = getListOfPlots(folderPath, testNames)
    if len(figs) == 0:
      print("Arithmeic test graph not made. No figures plotted")
      exit(1)
    else:
      print("saving "+str(len(figs))+" arithmetic test graphs")
    savePath = folderPath + "arithmeticGraphs.pdf"
    saveFigureList(figs, savePath)












