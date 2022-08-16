

#ifndef DENSENET201_SCNNOPS_H
#define DENSENET201_SCNNOPS_H
#include "scnn.h"
class Conv_1_128_7_7_32{
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
    float *cpuKernel;
    float *output;
    float *kernel;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_128_14_14_32{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 3;
    unsigned int TW = 7;
    unsigned int TC = 4;
    unsigned int TCS;
    float *cpuKernel;
    float *output;
    float *kernel;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_128_28_28_32{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 7;
    unsigned int TW = 7;
    unsigned int TC = 2;
    unsigned int TCS;
    float *cpuKernel;
    float *output;
    float *kernel;
    float * nnz;
    unsigned int TILES_EACH_CHANNEL;
    unsigned int * c_lens;
    char * ids;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile);
    float * forward(float *input);
};
class Conv_1_128_56_56_32{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int TH = 7;
    unsigned int TW = 7;
    unsigned int TC = 8;
    unsigned int TCS;
    float *cpuKernel;
    float *output;
    float *kernel;
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
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w);
    unsigned int *h_counter;
    float *forward(float *x);
};
#endif //DENSENET201_SCNNOPS_H
