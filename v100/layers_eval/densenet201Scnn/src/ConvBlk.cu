#include "../inc/convBlk.h"
#define MEASUE_CUSNN true
ConvBlk56::ConvBlk56(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                     string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight,bool sparse, int index,float *t1, float *t2) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w);
    conv_block_1_relu_cudnn.initialize(b,n1,h,w);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;
}
float * ConvBlk56::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out;
    float * conv_block_2_conv_out;
    conv_block_1_relu_out = conv_block_1_relu_cudnn.forward(conv_block_1_bn_out);
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    float temp_time;
    if(MEASUE_CUSNN){
        cudaEventRecord(event_start);
        conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        t1[index - 1] +=temp_time;
    }else{
        cudaEventRecord(event_start);
        conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        t2[index - 1] +=temp_time;
    }
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
ConvBlk28::ConvBlk28(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                     string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight, bool sparse,int index, float *t1, float *t2) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w);
    conv_block_1_relu_cudnn.initialize(b,n1,h,w);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;
}
float * ConvBlk28::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out;
    float * conv_block_2_conv_out;
    if(sparse){
        conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
    }else{
        conv_block_1_relu_out = conv_block_1_relu_cudnn.forward(conv_block_1_bn_out);
        conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
    }
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
ConvBlk14::ConvBlk14(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                     string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight, bool sparse, int index, float *t1, float *t2) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w);
    conv_block_1_relu_cudnn.initialize(b,n1,h,w);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;
}
float * ConvBlk14::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out;
    float * conv_block_2_conv_out;
    if(sparse){
        conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
    }else{
        conv_block_1_relu_out = conv_block_1_relu_cudnn.forward(conv_block_1_bn_out);
        conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
    }
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
ConvBlk7::ConvBlk7(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n1, unsigned int n2,
                   string bn0Weight, string conv1Weight, string bn1Weight, string conv2Weight,bool sparse, int index, float *t1, float *t2) {
    conv_block_0_bn.initialize(b,c,h,w,bn0Weight);
    conv_block_0_relu.initialize(b,c,h,w);
    conv_block_1_conv.initialize(b,c,h,w,n1,0,1,1,1,conv1Weight);
    conv_block_1_bn.initialize(b,n1,h,w,bn1Weight);
    conv_block_1_relu.initialize(b,n1,h,w);
    conv_block_1_relu_cudnn.initialize(b,n1,h,w);
    conv_block_2_conv_cudnn.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_2_conv.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv_block_concat.initialize(b,c,n2,h,w);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;
}
float * ConvBlk7::forward(float *input) {
    float * conv_block_0_bn_out = conv_block_0_bn.forward(input);
    float * conv_block_0_relu_out = conv_block_0_relu.forward(conv_block_0_bn_out);
    float * conv_block_1_conv_out = conv_block_1_conv.forward(conv_block_0_relu_out);
    float * conv_block_1_bn_out = conv_block_1_bn.forward(conv_block_1_conv_out);
    float * conv_block_1_relu_out;
    float * conv_block_2_conv_out;
    if(sparse){
        conv_block_1_relu_out = conv_block_1_relu.forward(conv_block_1_bn_out);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            conv_block_2_conv_out = conv_block_2_conv.forward(conv_block_1_relu_out);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
    }else{
        conv_block_1_relu_out = conv_block_1_relu_cudnn.forward(conv_block_1_bn_out);
        conv_block_2_conv_out = conv_block_2_conv_cudnn.forward(conv_block_1_relu_out);
    }
    float * conv_block_concat_out = conv_block_concat.forward(input,conv_block_2_conv_out);
    return conv_block_concat_out;
}
