

class BaseTestClass {
public:
  //the device id to run tests on
  // int deviceID;

  //framework to run and sample the kernel
  // TestRunner tester;

  //fields to prepare for kernel call
  int numBlocks;
  int blockSize;

	virtual void kernelSetup();
	virtual void runKernel();

}

