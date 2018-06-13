LIBS=-lnvidia-ml
FRAMEWORKFILES=arithmeticTests.cu testFramework.cu
HELPERFILES=testHelpers.cpp

all: arithmeticTest.out basePowerTest1.out basePowerTest2.out

arithmeticTest.out: runArithmeticTests.cu $(FRAMEWORKFILES) $(HELPERFILES)
	nvcc $< $(LIBS) -o $@ $(HELPERFILES)

basePowerTest1.out: runBasePowerTest1.cu $(FRAMEWORKFILES) $(HELPERFILES)
	nvcc $< $(LIBS) -o $@ $(HELPERFILES)

basePowerTest2.out: runBasePowerTest2.cu $(FRAMEWORKFILES) $(HELPERFILES)
	nvcc $< $(LIBS) -o $@ $(HELPERFILES)

