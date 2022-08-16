
#include "../inc/conv_and_pool.h"
#include "../inc/cudnnOps.h"
Conv_and_pool::Conv_and_pool(unsigned int b, unsigned int c, unsigned int h, unsigned int w,unsigned int n1,unsigned int n2,
                             string conv1Weight, string conv2Weight){
    conv1.initialize(b,c,h,w,n1,1,3,3,1,conv1Weight);
    conv2.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    relu1.initialize(b,n1,h,w);
    relu2.initialize(b,n2,h,w);
    pool.initialize(1,n2,h,w,1,3,3,CUDNN_POOLING_MAX,2);
}

float *Conv_and_pool::forward(float *input) {
    float *x = conv1.forward(input);
    x = relu1.forward(x);
    x = conv2.forward(x);
    x = relu2.forward(x);
    x = pool.forward(x);
    return x;
}
Conv_and_pool3::Conv_and_pool3(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                               unsigned int n1,unsigned int n2,unsigned int n3, string conv1Weight,
                               string conv2Weight, string conv3Weight){
    conv1.initialize(b,c,h,w,n1,1,3,3,1,conv1Weight);
    conv2.initialize(b,n1,h,w,n2,1,3,3,1,conv2Weight);
    conv3.initialize(b,n2,h,w,n3,1,3,3,1,conv3Weight);
    relu1.initialize(b,n1,h,w);
    relu2.initialize(b,n2,h,w);
    relu3.initialize(b,n3,h,w);
    pool.initialize(1,n3,h,w,1,3,3,CUDNN_POOLING_MAX,2);
}

float *Conv_and_pool3::forward(float *input) {
    float *x = conv1.forward(input);
    x = relu1.forward(x);
    x = conv2.forward(x);
    x = relu2.forward(x);
    x = conv3.forward(x);
    x = relu3.forward(x);
    x = pool.forward(x);
    return x;
}