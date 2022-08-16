#include "../inc/cudnnOps.h"
#include "../inc/cudnnOps.h"
__global__ void concateX(unsigned int b,unsigned int c1,unsigned int c2,unsigned int h,unsigned int w,float *x,float *z){
    unsigned int gId = threadIdx.x + blockIdx.x*1024;
    if(gId>=b*c1*h*w){
        return ;
    }
    for(unsigned int i=gId;i<b*c1*h*w;i+=8*1024){
        unsigned int id = i/c1;
        unsigned int cId = i%c1;
        z[id*(c1+c2)+cId] = x[id*c1+cId];
    }
}
__global__ void concateY(unsigned int b,unsigned int c1,unsigned int c2,unsigned int h,unsigned int w,float *y,float *z){
    unsigned int gId = threadIdx.x + blockIdx.x*1024;
    if(gId>=b*c2*h*w){
        return ;
    }
    for(unsigned int i=gId;i<b*c2*h*w;i+=8*1024){
        unsigned int id = i/c2;
        unsigned int cId = i%c2;
        z[id*(c1+c2)+cId+c1] = y[id*c2+cId];
    }
}
void Concate::initialize(unsigned int b, unsigned int c1, unsigned int c2, unsigned int h, unsigned int w) {
    B = b;
    C1 = c1;
    C2 = c2;
    H = h;
    W = w;
    cudaMalloc(&output,b*(c1+c2)*h*w*sizeof(float));
    for (int i = 0; i < 2; i ++){
        cudaStreamCreate(&streams[i]);
    }
}
float * Concate::forward(float *x, float *y) {
    concateX<<<8,1024,0,streams[0]>>>(B,C1,C2,H,W,x,output);
    concateY<<<8,1024,0,streams[1]>>>(B,C1,C2,H,W,y,output);
    return output;
}