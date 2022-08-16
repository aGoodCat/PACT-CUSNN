#include "../inc/common.h"
ConvBlk8::ConvBlk8(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string conv1Weight, string bn1Weight,
                   string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv1.initialize(B,C,H,W,512,0,1,1,1,conv1Weight);
    conv2.initialize(B,512,H,W,512,1,3,3,1,conv2Weight);
    conv3.initialize(B,512,H,W,2048,0,1,1,1,conv3Weight);
    bn1.initialize(B,512,H,W,bn1Weight);
    bn2.initialize(B,512,H,W,bn2Weight);
    bn3.initialize(B,2048,H,W,bn3Weight);
    add.initialize(B,2048,H,W);
    activation64.initialize(B,512,H,W);
    activation.initialize(B,2048,H,W);
}
float * ConvBlk8::forward(float *input){
    float *block2_1_conv = conv1.forward(input);
    float *block2_1_bn = bn1.forward(block2_1_conv);
    float *block2_1_relu = activation64.forward(block2_1_bn);

    float *block2_2_conv = conv2.forward(block2_1_relu);
    float *block2_2_bn = bn2.forward(block2_2_conv);
    float *block2_2_relu = activation64.forward(block2_2_bn);

    float *block2_3_conv = conv3.forward(block2_2_relu);
    float *block2_3_bn = bn3.forward(block2_3_conv);

    float *block2_add = add.forward(block2_3_bn,input);
    float *block2_out = activation.forward(block2_add);
    return block2_out;
}