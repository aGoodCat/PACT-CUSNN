
#ifndef RESNET152_COMMON_H
#define RESNET152_COMMON_H
#include <cudnn.h>
#include <stdio.h>
#include <cuda.h>
#include <malloc.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <sys/types.h>
#include <errno.h>
#include <vector>
#include <fstream>
#include <string>
using namespace std;
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<endl;
        exit(-1);
    }
}
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
void load_input(string input_path,unsigned int dataSize,float *input);
class Conv{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int PAD;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int R;
    unsigned int S;
    float *cpuKernel;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t biasDescriptor;
    float *output;
    float *kernel;
    float *bias;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
         unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float *forward(float *input);
};
class Pool{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t poolingCudnn;
    cudnnTensorDescriptor_t poolingInputDescriptor;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnTensorDescriptor_t poolingOutputDescriptor;
    float *output;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int pad,unsigned int windowH,unsigned int windowW,
         cudnnPoolingMode_t mode,unsigned int stride);
    float * forward(float *input);
};
class BatchNorm{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t batchNormCudnn;
    cudnnTensorDescriptor_t batchNormInputDescriptor;
    cudnnTensorDescriptor_t batchNormOutputDescriptor;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
    float *cpuKernel;
    float *scaleDev;
    float *shiftDev;
    float *meanDev;
    float *varDev;
    void initialize(unsigned int b,unsigned int c,unsigned int h, unsigned int w,string weightFile);
    float * forward(float *input);
    float *output;
};
class Activation{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t activationCudnn;
    cudnnActivationMode_t MODE;
    cudnnTensorDescriptor_t activationInputDescriptor;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t activationOutputDescriptor;
    float *output;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w);
    float *forward(float *input);
};
class Add{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnHandle_t addCudnn;
    cudnnTensorDescriptor_t addInputDescriptor;
    cudnnTensorDescriptor_t addOutputDescriptor;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w);
    float *forward(float *x,float *y);
};
class ConvBlk1{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv0;
    BatchNorm bn0;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk1(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv0Weight,string bn0Weight,
             string conv1Weight,string bn1Weight,string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk2{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk2(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv1Weight,string bn1Weight,
             string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk3{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv0;
    BatchNorm bn0;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk3(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv0Weight,string bn0Weight,
             string conv1Weight,string bn1Weight,string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk4{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk4(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv1Weight,string bn1Weight,
             string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk5{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv0;
    BatchNorm bn0;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk5(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv0Weight,string bn0Weight,
             string conv1Weight,string bn1Weight,string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk6{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk6(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv1Weight,string bn1Weight,
             string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk7{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv0;
    BatchNorm bn0;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk7(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv0Weight,string bn0Weight,
             string conv1Weight,string bn1Weight,string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
class ConvBlk8{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv conv1;
    BatchNorm bn1;
    Conv conv2;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Activation activation64;
    ConvBlk8(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv1Weight,string bn1Weight,string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
#endif //RESNET152_COMMON_H
