

#ifndef DENSENET201_CONVBLK_H
#define DENSENET201_CONVBLK_H
#include "cudnnOps.h"
#include "scnnOps.h"
class ConvBlk56{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    ConvBlk56(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
              string bn0Weight,string conv1Weight, string bn1Weight, string conv2Weight);
    BatchNorm conv_block_0_bn;
    Activation conv_block_0_relu;
    Conv conv_block_1_conv;
    BatchNorm conv_block_1_bn;
    Relu conv_block_1_relu;
    Conv conv_block_2_conv_cudnn;
    Conv_1_128_56_56_32 conv_block_2_conv;
    Concate conv_block_concat;
    float *forward(float *input);
};
class ConvBlk28{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    ConvBlk28(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
              string bn0Weight,string conv1Weight, string bn1Weight, string conv2Weight);
    BatchNorm conv_block_0_bn;
    Activation conv_block_0_relu;
    Conv conv_block_1_conv;
    BatchNorm conv_block_1_bn;
    Relu conv_block_1_relu;
    Conv conv_block_2_conv_cudnn;
    Conv_1_128_28_28_32 conv_block_2_conv;
    Concate conv_block_concat;
    float *forward(float *input);
};
class ConvBlk14{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    ConvBlk14(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,string bn0Weight,string conv1Weight, string bn1Weight, string conv2Weight);
    BatchNorm conv_block_0_bn;
    Activation conv_block_0_relu;
    Conv conv_block_1_conv;
    BatchNorm conv_block_1_bn;
    Activation conv_block_1_relu;
    Conv conv_block_2_conv_cudnn;
    Conv_1_128_14_14_32 conv_block_2_conv;
    Concate conv_block_concat;
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
    ConvBlk7(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,string bn0Weight,string conv1Weight, string bn1Weight, string conv2Weight);
    BatchNorm conv_block_0_bn;
    Activation conv_block_0_relu;
    Conv conv_block_1_conv;
    BatchNorm conv_block_1_bn;
    Activation conv_block_1_relu;
    Conv conv_block_2_conv_cudnn;
    Conv_1_128_7_7_32 conv_block_2_conv;
    Concate conv_block_concat;
    float *forward(float *input);
};
class PoolBlk{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int cOut;
    BatchNorm pool_bn;
    Activation pool_relu;
    Conv pool_conv;
    Pool pool;
    float *forward(float *input);
    PoolBlk(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n,string bn0Weight,string conv1Weight);
};
#endif //DENSENET201_CONVBLK_H