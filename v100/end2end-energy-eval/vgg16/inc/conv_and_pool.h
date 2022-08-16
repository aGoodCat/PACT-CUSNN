

#ifndef VGG16_CONV_AND_POOL_H
#define VGG16_CONV_AND_POOL_H
#include "./cudnnOps.h"
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
class Conv_and_pool3{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    Conv_and_pool3(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
                  unsigned int n3, string conv1Weight, string conv2Weight, string conv3Weight);
    Conv conv1;
    Conv conv2;
    Conv conv3;
    Activation relu1;
    Activation relu2;
    Activation relu3;
    Pool pool;
    float *forward(float *input);
};
#endif //VGG16_CONV_AND_POOL_H
