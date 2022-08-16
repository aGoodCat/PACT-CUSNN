

#ifndef DENSENET201_CUDNNOPS_H
#define DENSENET201_CUDNNOPS_H
#include "common.h"
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
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    float *output;
    float *kernel;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float *forward(float *input);
};
class FC{
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
class Concate{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C1;
    unsigned int B;
    unsigned int C2;
    float alpha = 1.0f;
    float beta = 1.0f;
    float *output;
    void initialize(unsigned int b,unsigned int c1,unsigned int c2,unsigned int h,unsigned int w);
    float *forward(float *x,float *y);
};
#endif //CPPSCNN_CUDNNOPS_H
