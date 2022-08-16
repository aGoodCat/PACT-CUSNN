#include "../inc/convBlk.h"
PoolBlk::PoolBlk(unsigned int b, unsigned int c, unsigned int h, unsigned int w, unsigned int n, string bn0Weight,
                 string conv1Weight) {
    pool_bn.initialize(b,c,h,w,bn0Weight);
    pool_relu.initialize(b,c,h,w);
    pool_conv.initialize(b,c,h,w,n,0,1,1,1,conv1Weight);
    pool.initialize(b,n,h,w,0,2,2,CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,2);
}
float * PoolBlk::forward(float *input) {
    float * pool_bn_out = pool_bn.forward(input);
    float * pool_relu_out = pool_relu.forward(pool_bn_out);
    float * pool_conv_out = pool_conv.forward(pool_relu_out);
    float * pool_out = pool.forward(pool_conv_out);
    return pool_out;
}