
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
} config_t;
