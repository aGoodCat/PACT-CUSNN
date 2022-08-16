#include "../inc/scnnOps.h"
void Relu::initialize(unsigned int b, unsigned int c, unsigned int h, unsigned int w) {
    B = b;
    C = c;
    H = h;
    W = w;
    cudaMalloc(&counter,1*sizeof(unsigned int));
    h_counter = new unsigned int[1];
}
float * Relu::forward(float *input) {
    sparse = false;
    cudaMemset(counter,0,sizeof(unsigned int));
    relu<<<68,1024>>>(input,B,C,H,W,68,counter);
    cudaMemcpy(h_counter,counter,1*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    if(float(h_counter[0])/float(B*C*H*W) >=0.6){
        sparse = true;
    }
    return input;
}