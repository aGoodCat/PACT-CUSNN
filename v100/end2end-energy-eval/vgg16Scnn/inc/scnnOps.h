
#ifndef VGG16SCNN_SCNNOPS_H
#define VGG16SCNN_SCNNOPS_H
#include "scnn.h"
class Conv_1_64_224_224_64{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 2;
    unsigned int TW = 7;
    unsigned int TC = 4;
    unsigned int TCS;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnTensorDescriptor_t biasDescriptor;
    float *cpuKernel;
    float *output;
    float *kernel;
    float *bias;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_128_112_112_128{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 2;
    unsigned int TW = 7;
    unsigned int TC = 4;
    unsigned int TCS;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnTensorDescriptor_t biasDescriptor;
    float *cpuKernel;
    float *output;
    float *kernel;
    float *bias;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_256_28_28_512{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 2;
    unsigned int TW = 7;
    unsigned int TC = 4;
    unsigned int TCS;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnTensorDescriptor_t biasDescriptor;
    float *cpuKernel;
    float *output;
    float *kernel;
    float *bias;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_256_56_56_256{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 2;
    unsigned int TW = 7;
    unsigned int TC = 4;
    unsigned int TCS;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnTensorDescriptor_t biasDescriptor;
    float *cpuKernel;
    float *output;
    float *kernel;
    float *bias;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_512_14_14_512{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 2;
    unsigned int TW = 7;
    unsigned int TC = 8;
    unsigned int TCS;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnTensorDescriptor_t biasDescriptor;
    float *cpuKernel;
    float *output;
    float *kernel;
    float *bias;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_512_28_28_512{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 2;
    unsigned int TW = 7;
    unsigned int TC = 4;
    unsigned int TCS;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnTensorDescriptor_t biasDescriptor;
    float *cpuKernel;
    float *output;
    float *kernel;
    float *bias;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Relu{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int *counter;
    bool sparse = false;
    unsigned int *h_counter;
    float sparse_threshold;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w, float threshold);
    float *forward(float *x);
};
#endif //VGG16SCNN_SCNNOPS_H
