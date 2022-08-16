#include "../inc/cudnnOps.h"
#include "../inc/cudnnOps.h"
__global__ void concate(unsigned int b,unsigned int c1,unsigned int c2,unsigned int h,unsigned int w,float *x,
                        float * y, float *z){
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id >=b*(c1+c2)*h*w){
        return ;
    }
    for(unsigned int i = id;i<b*h*w*(c1+c2);i+=gridDim.x*blockDim.x){
        unsigned int c = i % (c1 + c2);
        unsigned int c_id = i / (c1 + c2);
        if(c >= c1){
            float v = y[c_id * c2 + c - c1];
            z[c_id*(c1+c2)+c] = v;
        }else{
            float v = x[c_id * c1 + c];
            z[c_id*(c1+c2)+c] = v;
        }
    }
}
void Concate::initialize(unsigned int b, unsigned int c1, unsigned int c2, unsigned int h, unsigned int w) {
    B = b;
    C1 = c1;
    C2 = c2;
    H = h;
    W = w;
    cudaMalloc(&output,b*(c1+c2)*h*w*sizeof(float));
}
float * Concate::forward(float *x, float *y) {
    concate<<<84,1024>>>(B,C1,C2,H,W,x,y,output);
    return output;
}