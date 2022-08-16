#include "../inc/common.h"
ConvBlk3::ConvBlk3(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,512,0,1,1,2,conv0Weight);
    conv1.initialize(B,C,H,W,128,0,1,1,2,conv1Weight);
    conv2.initialize(B,128,H/2,W/2,128,1,3,3,1,conv2Weight);
    conv3.initialize(B,128,H/2,W/2,512,0,1,1,1,conv3Weight);
    bn0.initialize(B,512,H/2,W/2,bn0Weight);
    bn1.initialize(B,128,H/2,W/2,bn1Weight);
    bn2.initialize(B,128,H/2,W/2,bn2Weight);
    bn3.initialize(B,512,H/2,W/2,bn3Weight);
    add.initialize(B,512,H/2,W/2);
    activation64.initialize(B,128,H/2,W/2);
    activation.initialize(B,512,H/2,W/2);
}
float * ConvBlk3::forward(float *input){
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
/*int main(void){
    ConvBlk3 a(1,256,56,56,"../weights/conv3_block1_0_conv.bin","../weights/conv3_block1_0_bn.bin",
                          "../weights/conv3_block1_1_conv.bin","../weights/conv3_block1_1_bn.bin",
                          "../weights/conv3_block1_2_conv.bin","../weights/conv3_block1_2_bn.bin",
                          "../weights/conv3_block1_3_conv.bin","../weights/conv3_block1_3_bn.bin");
    float *input = (float *)malloc(256*56*56*sizeof(float));
    for(int i=0;i<256*56*56;++i){
        input[i] = 1.0f;
    }
    float *dInput;
    cudaMalloc(&dInput,256*56*56*sizeof(float));
    cudaMemcpy(dInput,input,256*56*56*sizeof(float),cudaMemcpyHostToDevice);
    float *dOutput = a.forward(dInput);
    float *output = (float *)malloc(512*28*28*sizeof(float));
    cudaMemcpy(output,dOutput,512*28*28*sizeof(float),cudaMemcpyDeviceToHost);
    float *pyOut = load_input("/home/lizhi/research/resnet152/convblk1.bin",64*56*256);
    cout<<output[0]<<endl;
    return 0;
}*/