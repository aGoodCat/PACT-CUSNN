#include "../inc/common.h"
ConvBlk5::ConvBlk5(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,1024,0,1,1,2,conv0Weight);
    conv1.initialize(B,C,H,W,256,0,1,1,2,conv1Weight);
    conv2.initialize(B,256,H/2,W/2,256,1,3,3,1,conv2Weight);
    conv3.initialize(B,256,H/2,W/2,1024,0,1,1,1,conv3Weight);
    bn0.initialize(B,1024,H/2,W/2,bn0Weight);
    bn1.initialize(B,256,H/2,W/2,bn1Weight);
    bn2.initialize(B,256,H/2,W/2,bn2Weight);
    bn3.initialize(B,1024,H/2,W/2,bn3Weight);
    add.initialize(B,1024,H/2,W/2);
    activation64.initialize(B,256,H/2,W/2);
    activation.initialize(B,1024,H/2,W/2);
}
float * ConvBlk5::forward(float *input){
    float *block3_1_conv = conv1.forward(input);
    float *block3_1_bn = bn1.forward(block3_1_conv);
    float *block3_1_relu = activation64.forward(block3_1_bn);

    float *block3_2_conv = conv2.forward(block3_1_relu);
    float *block3_2_bn = bn2.forward(block3_2_conv);
    float *block3_2_relu = activation64.forward(block3_2_bn);

    float *block3_3_conv = conv3.forward(block3_2_relu);
    float *block3_3_bn = bn3.forward(block3_3_conv);

    float *block3_0_conv = conv0.forward(input);
    float *block3_0_bn = bn0.forward(block3_0_conv);

    float *block3_add = add.forward(block3_3_bn,block3_0_bn);
    float *block3_out = activation.forward(block3_add);
    return block3_out;
}