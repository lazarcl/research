import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages

LINE_WIDTH = 1
SAVE_DPI = 1000
MAX_Y = 160

TEST_HEIGHT = 120
CONTROL_HEIGHT = 100
BP_HEIGHT = 30
TEST_LENGTH = 10000
CONTROL_LENGTH = 11500
# def makeFigure(self, file1, file2, testName):
# power1, time1 = dataLoader.getPowerAndTimeFromFile(file1)
# power2, time2 = dataLoader.getPowerAndTimeFromFile(file2)

xAxis1 = [i for i in range(TEST_LENGTH)]
yAxis1 = [CONTROL_HEIGHT for i in range(TEST_LENGTH - 1)] + [0]
# yAxis1 = [0] + [CONTROL_HEIGHT for i in range(TEST_LENGTH - 2)] + [0]

xAxis2 = [i for i in range(CONTROL_LENGTH)]
yAxis2 = [TEST_HEIGHT for i in range(CONTROL_LENGTH - 1)] + [0]
# yAxis2 = [0] + [TEST_HEIGHT for i in range(CONTROL_LENGTH - 2)] + [0]

xAxis3 = [i for i in range(CONTROL_LENGTH)]
yAxis3 = [BP_HEIGHT for i in range(CONTROL_LENGTH)]

f = pylab.figure()
ax = pylab.subplot(111)    
# ax.spines["top"].set_visible(False)    
# ax.spines["bottom"].set_visible(False)    
# ax.spines["right"].set_visible(False)    
# ax.spines["left"].set_visible(False)    

pylab.plot(xAxis2, yAxis2, "-r", label="Test Kernel", lw=LINE_WIDTH)
pylab.plot(xAxis1, yAxis1, "-b", label="Control Kernel", lw=LINE_WIDTH)
pylab.plot(xAxis3, yAxis3, "-g", label="Base Power", lw=LINE_WIDTH, linestyle='dashed')

#shading
ax.fill_between([i for i in range(TEST_LENGTH)], BP_HEIGHT, CONTROL_HEIGHT, color='lightsteelblue') #control power
ax.fill_between([i for i in range(CONTROL_LENGTH)], CONTROL_HEIGHT, TEST_HEIGHT, color='rosybrown') #marginal test pow top
ax.fill_between([i for i in range(TEST_LENGTH, CONTROL_LENGTH)], BP_HEIGHT, TEST_HEIGHT, color='rosybrown') #marginal test pow right
ax.fill_between([i for i in range(CONTROL_LENGTH)], 0, BP_HEIGHT, color='mediumseagreen') #base pow


# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
x_l,x_r = ax.get_xlim()
ax.set_xlim(0, x_r)



pylab.xlabel('time(ms)')
pylab.ylabel('power(W)')
pylab.title("Ideal Base Power Energy Consumption in Kernel Runs")

pylab.legend(loc="upper right")
pylab.ylim(0, MAX_Y)



# savePath = "~/Desktop/"
savePath = "figures/"
pdfName = "basePowTheory"
#save to seperate file

pdf = PdfPages(savePath + pdfName + ".pdf")
pdf.savefig(f, dpi=SAVE_DPI, bbox_inches='tight')
pdf.close()

# f.savefig(savePath + pdfName + ".pdf", bbox_inches='tight')



