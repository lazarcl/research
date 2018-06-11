import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages

'''
helpful styling guide:

http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
'''

##constants
SAVE_DPI = 1000
LINE_WIDTH = 0.4
MAX_Y = 80


#given filepath, return list of power data as FPs
def getPowerFromFile(filePath):
	colnames = ['power', 'temp', 'time', 'totalT', 'totalSamples']
	data = pandas.read_csv(filePath, names=colnames, encoding='utf-8')

	power = data.power.tolist()[1:]
	power = [float(power[i]) for i in range(len(power))]
	return power

#name of test to make graph for
#  ex: FMAFP32, FMAFP64, AddInt32, MultFP32...
def makeFigure(testStr):
	power1 = getPowerFromFile("data/output" + testStr + "_1.csv")
	power2 = getPowerFromFile("data/output" + testStr + "_2.csv")

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
	pylab.title(testStr + " Kernel Runs")

	pylab.legend(loc="lower right")
	pylab.ylim(0, MAX_Y)

	#save to seperate file
	# f.savefig("data/"+ testStr + ".pdf", bbox_inches='tight')

	return f

#input: list of test strings to make graphs for
#  ex: ["FMAFP32", "AddInt32"]
#output: list of plots. One plot per input element
def getListOfPlots(listOfTests):
	out = []
	for test in listOfTests:
		out.append(makeFigure(test))
	return out

#given a file name to save to, and a list of figures, save to one pdf
def saveFigureList(figs, filePath):
	pp = PdfPages(filePath)
	for fig in figs:
		pp.savefig(fig, dpi=SAVE_DPI)
	pp.close()

testNames = ["AddFP32", "AddFP64", "AddInt32", "FMAFP32", "FMAFP64", "MultFP32", "MultFP64", "MultInt32"]
figs = getListOfPlots(testNames)
saveFigureList(figs, "data/arithmeticGraphs.pdf")












