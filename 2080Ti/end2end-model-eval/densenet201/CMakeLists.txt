cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -arch=sm_75")
project(densenet201 LANGUAGES CXX CUDA)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories($ENV{cuda_path})
link_directories($ENV{cuda_lib_path})
add_executable(test inc/common.h inc/cudnnOps.h inc/convBlk.h src/common.cpp src/FC.cu src/Concate.cu src/Add.cu src/Conv.cu src/BatchNorm.cu src/Pool.cu src/Activation.cu
        src/PoolBlk.cu src/ConvBlk.cu src/main.cu)
target_link_libraries(test ${CUDA_LIBRARIES} cudnn nvidia-ml)
