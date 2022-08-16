#include "../inc/scnnOps.h"
void Conv_1_128_28_28_32::initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                                     unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile) {
    this->B = b;
    this->C = c;
    this->H = h;
    this->W = w;
    this->N = n;
    this->hOut = h;
    this->wOut = w;
    unsigned int kernelSize = 3*3*C*N;//kernel + bias
    this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
    try{
        load_input(weightFile,kernelSize,cpuKernel);
    }catch (const char* msg) {
        cerr << msg << endl;
    }
    float *temp_kernel = (float *)malloc(kernelSize*sizeof(float));
    for(unsigned int i=0;i<N;++i){
        for(unsigned int l=0;l<C;++l){
            for(unsigned int j=0;j<3;++j){
                for(unsigned int k=0;k<3;++k){
                    temp_kernel[l*N*3*3+j*3*N+k*N+i] = cpuKernel[i*3*3*C+l*9+j*3+k];
                }
            }
        }
    }
    cudaMalloc(&kernel,9*C*N*sizeof(float));
    cudaMemcpy(kernel,temp_kernel,9*C*N*sizeof(float),cudaMemcpyHostToDevice);
    free(temp_kernel);
    free(cpuKernel);
    TILES_EACH_CHANNEL = ((H-1)/TH+1)*((W-1)/TW+1);
    TCS = (C-1)/TC + 1;
    cudaMalloc(&nnz,b*c*TILES_EACH_CHANNEL*(TH+3-1)*(TW+3-1)*sizeof(float));
    cudaMalloc(&ids,b*c*TILES_EACH_CHANNEL*(TH+3-1)*(TW+3-1)*sizeof(char));
    cudaMalloc(&c_lens,b*c*TILES_EACH_CHANNEL*sizeof(unsigned int));
    cudaMalloc(&output,b*n*h*w*sizeof(float));
}
float * Conv_1_128_28_28_32::forward(float *input) {
    cudaMemset(output, 0, B*N*hOut*wOut*sizeof(float));
    cudaMemset(c_lens, 0, B*C*TILES_EACH_CHANNEL*sizeof(unsigned int));
    conv_1_128_28_28_32_transform<<<(B*C*H*W-1)/512+1,512>>>(input,this->nnz,this->ids,this->c_lens);
    conv_1_128_28_28_32_conv2d<<<B*TCS*TILES_EACH_CHANNEL,N>>>(this->nnz,this->c_lens,this->ids,this->kernel,this->output);

    //chkerr(cudaGetLastError());
    //chkerr(cudaDeviceSynchronize());
    return output;
}
