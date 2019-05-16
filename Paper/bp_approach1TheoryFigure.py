import pylab
import pandas
from matplotlib.backends.backend_pdf import PdfPages
#changing font size: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
#answer by binaryfunt
LINE_WIDTH = 1
SAVE_DPI = 1000
MAX_Y = 180
RED = '#B60000'
BLUE = '#0037FB'
GREEN = '#1E7C20'

TEST_HEIGHT = 120
CONTROL_HEIGHT = 100
BP_HEIGHT = 30
TEST_LENGTH = 10000
CONTROL_LENGTH = 11500


# def makeFigure(self, file1, file2, testName):
# power1, time1 = dataLoader.getPowerAndTimeFromFile(file1)
# power2, time2 = dataLoader.getPowerAndTimeFromFile(file2)

xAxis1 = [i for i in range(TEST_LENGTH)]
yAxis1 = [TEST_HEIGHT for i in range(TEST_LENGTH - 1)] + [BP_HEIGHT]
# yAxis1 = [0] + [CONTROL_HEIGHT for i in range(TEST_LENGTH - 2)] + [0]

xAxis2 = [i for i in range(CONTROL_LENGTH)]
yAxis2 = [CONTROL_HEIGHT for i in range(CONTROL_LENGTH - 1)] + [BP_HEIGHT]
# yAxis2 = [0] + [TEST_HEIGHT for i in range(CONTROL_LENGTH - 2)] + [0]

xAxis3 = [i for i in range(CONTROL_LENGTH)]
yAxis3 = [BP_HEIGHT for i in range(CONTROL_LENGTH)]


# xAxis1 = [i for i in range(11500)]
# yAxis1 = [100 for i in range(11500 - 1)] + [0]

# xAxis2 = [i for i in range(10000)]
# yAxis2 = [120 for i in range(10000 - 1)] + [0]


f = pylab.figure()
ax = pylab.subplot(111)    
# ax.spines["top"].set_visible(False)    
# ax.spines["bottom"].set_visible(False)    
# ax.spines["right"].set_visible(False)    
# ax.spines["left"].set_visible(False)    

pylab.plot(xAxis1, yAxis1, RED, label="Fully Utillized Kernel", lw=LINE_WIDTH)
pylab.plot(xAxis2, yAxis2, BLUE, label="Underutilized Kernel", lw=LINE_WIDTH)
pylab.plot(xAxis3, yAxis3, GREEN, label="Base Power", lw=LINE_WIDTH, linestyle='dashed')

ax.fill_between([i for i in range(TEST_LENGTH)], BP_HEIGHT, TEST_HEIGHT, facecolor='none', edgecolor=RED, hatch='/////') #control power
ax.fill_between([i for i in range(CONTROL_LENGTH)], BP_HEIGHT, CONTROL_HEIGHT, facecolor='none', edgecolor=BLUE, hatch='\\\\\\\\\\') #marginal test pow top
ax.fill_between([i for i in range(CONTROL_LENGTH)], 0, BP_HEIGHT, facecolor='none', edgecolor=GREEN, hatch="...") #base pow


# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
x_l,x_r = ax.get_xlim()
ax.set_xlim(0, x_r)


pylab.xlabel('time (ms)')
pylab.ylabel('power (W)')
# pylab.title("Measuring Base Power with Underutilized Kernels")

pylab.legend(loc="upper right")
pylab.ylim(0, MAX_Y)



# savePath = "~/Desktop/"
savePath = "figures/"
pdfName = "bpApproach1Theory"
#save to seperate file

pdf = PdfPages(savePath + pdfName + ".pdf")
pdf.savefig(f, dpi=SAVE_DPI, bbox_inches='tight')
pdf.close()

# f.savefig(savePath + pdfName + ".pdf", bbox_inches='tight')



