cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cppScnn LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -arch=sm_70")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(/usr/local/cuda/include)
include_directories($ENV{cuda_path})
link_directories($ENV{cuda_lib_path})
add_executable(test inc/common.h inc/scnn.h inc/cudnnOps.h inc/scnnOps.h inc/convBlk.h src/Relu.cu src/relu.cu src/common.cpp
        src/Conv_1_64_56_56_64.cu src/Conv_1_128_28_28_128.cu
        src/Conv_1_256_14_14_256.cu src/Conv_1_512_7_7_512.cu src/Add.cu src/Conv.cu src/BatchNorm.cu src/Pool.cu src/Activation.cu
        src/ConvBlk1.cu src/main.cu)
target_link_libraries(test ${CUDA_LIBRARIES} cudnn nvidia-ml)
