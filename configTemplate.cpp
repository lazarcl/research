
//Template for test config file. Update as neccisarry and save as 'config.cpp'


//config struct for tests.
//both deviceID and deviceName are neccisarry because of the 
//  difference between nvml and cuda GPU indexing
struct {
	//id of the device to poll power data of.
	//find by looking at GPU id's from 'nvidia-smi' command
  int deviceID = 0;

  //name of the graphics card to run kernels on.
  //look up with 'nvidia-smi' command
  const char *deviceName = "Tesla K20c";

  //iteration amount for each arithmetic test
  int AddFP32_iter = 2000000;
  int AddFP64_iter = 2000000 / 3;
  int AddInt32_iter = 2000000;
  int MultFP32_iter = 2000000;
  int MultFP64_iter = 2000000 / 3;
  int MultInt32_iter = 2000000;
  int FMAFP32_iter = 800000*2;
  int FMAFP64_iter = 800000;

  //iteration amount for each test in first approach to base power
  int basePow1_iter = 2000000;

  //iteration amount for each test in 2nd approach to base power
  int basePow2_iter = 40000000;


} config_t;
