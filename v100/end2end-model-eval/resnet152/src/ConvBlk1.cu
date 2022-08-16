#include "../inc/common.h"
// conv2_block1
ConvBlk1::ConvBlk1(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight){
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,256,0,1,1,1,conv0Weight);
    conv1.initialize(B,C,H,W,64,0,1,1,1,conv1Weight);
    conv2.initialize(B,C,H,W,64,1,3,3,1,conv2Weight);
    conv3.initialize(B,C,H,W,256,0,1,1,1,conv3Weight);
    bn0.initialize(B,256,H,W,bn0Weight);
    bn1.initialize(B,64,H,W,bn1Weight);
    bn2.initialize(B,64,H,W,bn2Weight);
    bn3.initialize(B,256,H,W,bn3Weight);
    add.initialize(B,256,H,W);
    activation64.initialize(B,64,H,W);
    activation.initialize(B,256,H,W);
}
float * ConvBlk1::forward(float *input){
    float *block1_1_conv = conv1.forward(input);
    float *block1_1_bn = bn1.forward(block1_1_conv);
    float *block1_1_relu = activation64.forward(block1_1_bn);

    float *block1_2_conv = conv2.forward(block1_1_relu);
    float *block1_2_bn = bn2.forward(block1_2_conv);
    float *block1_2_relu = activation64.forward(block1_2_bn);

    float *block1_3_conv = conv3.forward(block1_2_relu);
    float *block1_3_bn = bn3.forward(block1_3_conv);

    float *block1_0_conv = conv0.forward(input);
    float *block1_0_bn = bn0.forward(block1_0_conv);

    float *block1_add = add.forward(block1_3_bn,block1_0_bn);
    float *relu = activation.forward(block1_add);
    return relu;
}
/*int main(void){
    ConvBlk1 a(1,64,56,56,"/home/lizhi/research/resnet152/weights/conv2_block1_0_conv.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_0_bn.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_1_conv.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_1_bn.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_2_conv.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_2_bn.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_3_conv.bin",
               "/home/lizhi/research/resnet152/weights/conv2_block1_3_bn.bin");
    float *input = (float *)malloc(64*56*56*sizeof(float));
    for(int i=0;i<64*56*56;++i){
        input[i] = 1.0f;
    }
    float *dInput;
    cudaMalloc(&dInput,64*56*56*sizeof(float));
    cudaMemcpy(dInput,input,64*56*56*sizeof(float),cudaMemcpyHostToDevice);
    float *dOutput = a.forward(dInput);
    float *output = (float *)malloc(256*56*56*sizeof(float));
    cudaMemcpy(output,dOutput,256*56*56*sizeof(float),cudaMemcpyDeviceToHost);
    float *pyOut = load_input("/home/lizhi/research/resnet152/convblk1.bin",64*56*256);
    cout<<pyOut[1]<<" "<<output[1]<<endl;
    float diff = 0.0f;
    for(int i=0;i<56*56*256;i++){
        diff += pyOut[i] - output[i];
    }
    cout<<diff<<endl;
    return 0;
}*/