
#ifndef VGG16_CONV_AND_POOL_H
#define VGG16_CONV_AND_POOL_H
#include "./cudnnOps.h"
#include "./scnnOps.h"
class Conv_and_pool{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv_and_pool(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
             string conv1Weight, string conv2Weight);
    Conv conv1;
    Conv conv2;
    Activation relu1;
    Activation relu2;
    Pool pool;
    float *forward(float *input);
};
class Conv_and_pool4{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv_and_pool4(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
                   unsigned int n3, unsigned int n4,string conv1Weight, string conv2Weight, string conv3Weight,string conv4Weight);
    Conv conv1;
    Conv conv2;
    Conv conv3;
    Conv conv4;
    Activation relu1;
    Activation relu2;
    Activation relu3;
    Activation relu4;
    Pool pool;
    float *forward(float *input);
};
class Conv_and_pool4_28{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv_and_pool4_28(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
                      unsigned int n3, unsigned int n4,string conv1Weight, string conv2Weight, string conv3Weight,string conv4Weight);
    Conv conv1;
    Conv conv2;
    Conv_1_512_28_28_512 conv3;
    Conv_1_512_28_28_512 conv4;

    Conv conv3_cudnn;
    Conv conv4_cudnn;
    Activation relu1;
    Relu relu2;
    Relu relu3;
    Activation relu4;
    Pool pool;
    float *forward(float *input);
};

class Conv_and_pool4_14{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv_and_pool4_14(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
                      unsigned int n3, unsigned int n4,string conv1Weight, string conv2Weight, string conv3Weight,string conv4Weight);
    Conv_1_512_14_14_512 conv1;
    Conv_1_512_14_14_512 conv2;
    Conv_1_512_14_14_512 conv3;
    Conv_1_512_14_14_512 conv4;

    Conv conv2_cudnn;
    Conv conv3_cudnn;
    Conv conv4_cudnn;

    Relu relu1;
    Relu relu2;
    Relu relu3;
    Activation relu4;
    Pool pool;
    float *forward(float *input);
};
#endif //VGG16_CONV_AND_POOL_H
