#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define NVML_DEV_t nvmlDevice_t

#define NVML_CHK(X) NVML_CHK_IMPL(X,__LINE__)

#define NVML_CHK_IMPL(X, line) \
    if (NVML_SUCCESS != (X)) { std::cout << "Error in line " << line << ". Error code is " << X <<"\n";  std::exit(EXIT_FAILURE);}

#define NVML_INIT NVML_CHK(nvmlInit());
#define NVML_GET_HANDLE(devid, handle_ptr) NVML_CHK(nvmlDeviceGetHandleByIndex(devid,handle_ptr));
#define NVML_MEASURE(devid,result_ull_ptr) NVML_CHK(nvmlDeviceGetTotalEnergyConsumption(devid, result_ull_ptr));
