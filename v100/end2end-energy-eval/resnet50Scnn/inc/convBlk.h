
#ifndef CPPSCNN_CONVBLK_H
#define CPPSCNN_CONVBLK_H
#include "cudnnOps.h"
#include "scnnOps.h"
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
    Conv_1_64_56_56_64 conv2;
    //Conv conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    //Conv conv2;
    Conv_1_64_56_56_64 conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    Conv_1_128_28_28_128 conv2;
    //Conv conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    Conv_1_128_28_28_128 conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    Conv_1_256_14_14_256 conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    Conv_1_256_14_14_256 conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    Conv_1_512_7_7_512 conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
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
    Conv_1_512_7_7_512 conv2;
    Conv conv2_cudnn;
    BatchNorm bn2;
    Conv conv3;
    BatchNorm bn3;
    Add add;
    Activation activation;
    Relu activation64;
    Activation activation64_cudnn;
    ConvBlk8(unsigned int b,unsigned int c,unsigned int h,unsigned int w,string conv1Weight,string bn1Weight,string conv2Weight,string bn2Weight,string conv3Weight,string bn3Weight);
    float *forward(float *input);
};
#endif //CPPSCNN_CONVBLK_H
