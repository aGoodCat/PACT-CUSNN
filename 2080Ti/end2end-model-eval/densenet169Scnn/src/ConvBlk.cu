#include "../inc/convBlk.h"
ConvBlk56::ConvBlk56(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                     string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w,0.6);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
}
float * ConvBlk56::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
    float * conv_block_2_conv_out;
    if(conv_block_1_relu.sparse){
        conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
    }else {
        conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
    }
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
ConvBlk28::ConvBlk28(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                     string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w,0.01);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
}
float * ConvBlk28::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
    float * conv_block_2_conv_out;
    if(conv_block_1_relu.sparse){
        conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
    }else{
        conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
    }
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
ConvBlk14::ConvBlk14(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                     string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
}
float * ConvBlk14::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
    float * conv_block_2_conv_out;
    conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
ConvBlk7::ConvBlk7(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                   string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
}
float * ConvBlk7::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
    float * conv_block_2_conv_out;
    conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}