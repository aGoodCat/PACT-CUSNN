#include "../inc/convBlk.h"
#include "../inc/nvml_helper.hpp"

int main(int argc,char *argv[]){
    float *input = (float *)malloc(224*224*3*sizeof(float));
    string imagePath = argv[1];
    //load_input(imagePath,3*224*224,input);
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    float *t1 = new float[100];
    float *t2 = new float[100];
    Conv conv1_conv;
    conv1_conv.initialize(1,3,224,224,64,3,7,7,2,"../../../weights/resnet152/weights/conv1_conv.bin");
    BatchNorm conv1_bn;
    conv1_bn.initialize(1,64,112,112,"../../../weights/resnet152/weights/conv1_bn.bin");
    Activation conv1_relu;
    conv1_relu.initialize(1,64,112,112);
    Pool conv1_max_pool;
    conv1_max_pool.initialize(1,64,112,112,1,3,3,CUDNN_POOLING_MAX,2);

    ConvBlk1 conv2_block1(1,64,56,56,"../../../weights/resnet152/weights/conv2_block1_0_conv.bin","../../../weights/resnet152/weights/conv2_block1_0_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block1_1_conv.bin","../../../weights/resnet152/weights/conv2_block1_1_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block1_2_conv.bin","../../../weights/resnet152/weights/conv2_block1_2_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block1_3_conv.bin","../../../weights/resnet152/weights/conv2_block1_3_bn.bin",
                          true,1,t1,t2);
    ConvBlk2 conv2_block2(1,256,56,56,"../../../weights/resnet152/weights/conv2_block2_1_conv.bin","../../../weights/resnet152/weights/conv2_block2_1_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block2_2_conv.bin","../../../weights/resnet152/weights/conv2_block2_2_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block2_3_conv.bin","../../../weights/resnet152/weights/conv2_block2_3_bn.bin",
                          true,2,t1,t2);
    ConvBlk2 conv2_block3(1,256,56,56,"../../../weights/resnet152/weights/conv2_block3_1_conv.bin","../../../weights/resnet152/weights/conv2_block3_1_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block3_2_conv.bin","../../../weights/resnet152/weights/conv2_block3_2_bn.bin",
                          "../../../weights/resnet152/weights/conv2_block3_3_conv.bin","../../../weights/resnet152/weights/conv2_block3_3_bn.bin",
                          true,3,t1,t2);
    ConvBlk3 conv3_block1(1,256,56,56,"../../../weights/resnet152/weights/conv3_block1_0_conv.bin","../../../weights/resnet152/weights/conv3_block1_0_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block1_1_conv.bin","../../../weights/resnet152/weights/conv3_block1_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block1_2_conv.bin","../../../weights/resnet152/weights/conv3_block1_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block1_3_conv.bin","../../../weights/resnet152/weights/conv3_block1_3_bn.bin",
                          true,4,t1,t2);
    ConvBlk4 conv3_block2(1,512,28,28,"../../../weights/resnet152/weights/conv3_block2_1_conv.bin","../../../weights/resnet152/weights/conv3_block2_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block2_2_conv.bin","../../../weights/resnet152/weights/conv3_block2_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block2_3_conv.bin","../../../weights/resnet152/weights/conv3_block2_3_bn.bin",
                          true,5,t1,t2);
    ConvBlk4 conv3_block3(1,512,28,28,"../../../weights/resnet152/weights/conv3_block3_1_conv.bin","../../../weights/resnet152/weights/conv3_block3_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block3_2_conv.bin","../../../weights/resnet152/weights/conv3_block3_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block3_3_conv.bin","../../../weights/resnet152/weights/conv3_block3_3_bn.bin",
                          true,6,t1,t2);
    ConvBlk4 conv3_block4(1,512,28,28,"../../../weights/resnet152/weights/conv3_block4_1_conv.bin","../../../weights/resnet152/weights/conv3_block4_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block4_2_conv.bin","../../../weights/resnet152/weights/conv3_block4_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block4_3_conv.bin","../../../weights/resnet152/weights/conv3_block4_3_bn.bin",
                          true,7,t1,t2);
    ConvBlk4 conv3_block5(1,512,28,28,"../../../weights/resnet152/weights/conv3_block5_1_conv.bin","../../../weights/resnet152/weights/conv3_block5_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block5_2_conv.bin","../../../weights/resnet152/weights/conv3_block5_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block5_3_conv.bin","../../../weights/resnet152/weights/conv3_block5_3_bn.bin",
                          true,8,t1,t2);
    ConvBlk4 conv3_block6(1,512,28,28,"../../../weights/resnet152/weights/conv3_block6_1_conv.bin","../../../weights/resnet152/weights/conv3_block6_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block6_2_conv.bin","../../../weights/resnet152/weights/conv3_block6_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block6_3_conv.bin","../../../weights/resnet152/weights/conv3_block6_3_bn.bin",
                          true,9,t1,t2);
    ConvBlk4 conv3_block7(1,512,28,28,"../../../weights/resnet152/weights/conv3_block7_1_conv.bin","../../../weights/resnet152/weights/conv3_block7_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block7_2_conv.bin","../../../weights/resnet152/weights/conv3_block7_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block7_3_conv.bin","../../../weights/resnet152/weights/conv3_block7_3_bn.bin",
                          true,10,t1,t2);
    ConvBlk4 conv3_block8(1,512,28,28,"../../../weights/resnet152/weights/conv3_block8_1_conv.bin","../../../weights/resnet152/weights/conv3_block8_1_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block8_2_conv.bin","../../../weights/resnet152/weights/conv3_block8_2_bn.bin",
                          "../../../weights/resnet152/weights/conv3_block8_3_conv.bin","../../../weights/resnet152/weights/conv3_block8_3_bn.bin",
                          true,11,t1,t2);

    ConvBlk5 conv4_block1(1,512,28,28,"../../../weights/resnet152/weights/conv4_block1_0_conv.bin","../../../weights/resnet152/weights/conv4_block1_0_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block1_1_conv.bin","../../../weights/resnet152/weights/conv4_block1_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block1_2_conv.bin","../../../weights/resnet152/weights/conv4_block1_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block1_3_conv.bin","../../../weights/resnet152/weights/conv4_block1_3_bn.bin",
                          true,12,t1,t2);
    ConvBlk6 conv4_block2(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block2_1_conv.bin","../../../weights/resnet152/weights/conv4_block2_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block2_2_conv.bin","../../../weights/resnet152/weights/conv4_block2_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block2_3_conv.bin","../../../weights/resnet152/weights/conv4_block2_3_bn.bin",
                          true,13,t1,t2);
    ConvBlk6 conv4_block3(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block3_1_conv.bin","../../../weights/resnet152/weights/conv4_block3_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block3_2_conv.bin","../../../weights/resnet152/weights/conv4_block3_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block3_3_conv.bin","../../../weights/resnet152/weights/conv4_block3_3_bn.bin",
                          true,14,t1,t2);
    ConvBlk6 conv4_block4(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block4_1_conv.bin","../../../weights/resnet152/weights/conv4_block4_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block4_2_conv.bin","../../../weights/resnet152/weights/conv4_block4_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block4_3_conv.bin","../../../weights/resnet152/weights/conv4_block4_3_bn.bin",
                          true,15,t1,t2);
    ConvBlk6 conv4_block5(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block5_1_conv.bin","../../../weights/resnet152/weights/conv4_block5_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block5_2_conv.bin","../../../weights/resnet152/weights/conv4_block5_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block5_3_conv.bin","../../../weights/resnet152/weights/conv4_block5_3_bn.bin",
                          true,16,t1,t2);
    ConvBlk6 conv4_block6(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block6_1_conv.bin","../../../weights/resnet152/weights/conv4_block6_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block6_2_conv.bin","../../../weights/resnet152/weights/conv4_block6_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block6_3_conv.bin","../../../weights/resnet152/weights/conv4_block6_3_bn.bin",
                          true,17,t1,t2);
    ConvBlk6 conv4_block7(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block7_1_conv.bin","../../../weights/resnet152/weights/conv4_block7_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block7_2_conv.bin","../../../weights/resnet152/weights/conv4_block7_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block7_3_conv.bin","../../../weights/resnet152/weights/conv4_block7_3_bn.bin",
                          true,18,t1,t2);
    ConvBlk6 conv4_block8(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block8_1_conv.bin","../../../weights/resnet152/weights/conv4_block8_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block8_2_conv.bin","../../../weights/resnet152/weights/conv4_block8_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block8_3_conv.bin","../../../weights/resnet152/weights/conv4_block8_3_bn.bin",
                          true,19,t1,t2);
    ConvBlk6 conv4_block9(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block9_1_conv.bin","../../../weights/resnet152/weights/conv4_block9_1_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block9_2_conv.bin","../../../weights/resnet152/weights/conv4_block9_2_bn.bin",
                          "../../../weights/resnet152/weights/conv4_block9_3_conv.bin","../../../weights/resnet152/weights/conv4_block9_3_bn.bin",
                          true,20,t1,t2);
    ConvBlk6 conv4_block10(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block10_1_conv.bin","../../../weights/resnet152/weights/conv4_block10_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block10_2_conv.bin","../../../weights/resnet152/weights/conv4_block10_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block10_3_conv.bin","../../../weights/resnet152/weights/conv4_block10_3_bn.bin",
                           true,21,t1,t2);
    ConvBlk6 conv4_block11(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block11_1_conv.bin","../../../weights/resnet152/weights/conv4_block11_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block11_2_conv.bin","../../../weights/resnet152/weights/conv4_block11_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block11_3_conv.bin","../../../weights/resnet152/weights/conv4_block11_3_bn.bin",
                           true,22,t1,t2);
    ConvBlk6 conv4_block12(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block12_1_conv.bin","../../../weights/resnet152/weights/conv4_block12_1_bn.bin","../../../weights/resnet152/weights/conv4_block12_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block12_2_bn.bin","../../../weights/resnet152/weights/conv4_block12_3_conv.bin","../../../weights/resnet152/weights/conv4_block12_3_bn.bin",
                           true,23,t1,t2);
    ConvBlk6 conv4_block13(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block13_1_conv.bin","../../../weights/resnet152/weights/conv4_block13_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block13_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block13_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block13_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block13_3_bn.bin",
                           true,24,t1,t2);
    ConvBlk6 conv4_block14(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block14_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block14_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block14_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block14_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block14_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block14_3_bn.bin",
                           true,25,t1,t2);
    ConvBlk6 conv4_block15(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block15_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block15_1_bn.bin","../../../weights/resnet152/weights/conv4_block15_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block15_2_bn.bin","../../../weights/resnet152/weights/conv4_block15_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block15_3_bn.bin",
                           true,26,t1,t2);
    ConvBlk6 conv4_block16(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block16_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block16_1_bn.bin","../../../weights/resnet152/weights/conv4_block16_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block16_2_bn.bin","../../../weights/resnet152/weights/conv4_block16_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block16_3_bn.bin",
                           true,27,t1,t2);
    ConvBlk6 conv4_block17(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block17_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block17_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block17_2_conv.bin","../../../weights/resnet152/weights/conv4_block17_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block17_3_conv.bin","../../../weights/resnet152/weights/conv4_block17_3_bn.bin",
                           true,28,t1,t2);
    ConvBlk6 conv4_block18(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block18_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block18_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block18_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block18_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block18_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block18_3_bn.bin",
                           true,29,t1,t2);
    ConvBlk6 conv4_block19(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block19_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block19_1_bn.bin","../../../weights/resnet152/weights/conv4_block19_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block19_2_bn.bin","../../../weights/resnet152/weights/conv4_block19_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block19_3_bn.bin",
                           true,30,t1,t2);
    ConvBlk6 conv4_block20(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block20_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block20_1_bn.bin","../../../weights/resnet152/weights/conv4_block20_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block20_2_bn.bin","../../../weights/resnet152/weights/conv4_block20_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block20_3_bn.bin",
                           true,31,t1,t2);
    ConvBlk6 conv4_block21(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block21_1_conv.bin","../../../weights/resnet152/weights/conv4_block21_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block21_2_conv.bin","../../../weights/resnet152/weights/conv4_block21_2_bn.bin","../../../weights/resnet152/weights/conv4_block21_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block21_3_bn.bin",
                           true,32,t1,t2);
    ConvBlk6 conv4_block22(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block22_1_conv.bin","../../../weights/resnet152/weights/conv4_block22_1_bn.bin","../../../weights/resnet152/weights/conv4_block22_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block22_2_bn.bin","../../../weights/resnet152/weights/conv4_block22_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block22_3_bn.bin",
                           true,33,t1,t2);
    ConvBlk6 conv4_block23(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block23_1_conv.bin","../../../weights/resnet152/weights/conv4_block23_1_bn.bin","../../../weights/resnet152/weights/conv4_block23_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block23_2_bn.bin","../../../weights/resnet152/weights/conv4_block23_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block23_3_bn.bin",
                           true,34,t1,t2);
    ConvBlk6 conv4_block24(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block24_1_conv.bin","../../../weights/resnet152/weights/conv4_block24_1_bn.bin","../../../weights/resnet152/weights/conv4_block24_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block24_2_bn.bin","../../../weights/resnet152/weights/conv4_block24_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block24_3_bn.bin",
                           true,35,t1,t2);
    ConvBlk6 conv4_block25(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block25_1_conv.bin","../../../weights/resnet152/weights/conv4_block25_1_bn.bin","../../../weights/resnet152/weights/conv4_block25_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block25_2_bn.bin","../../../weights/resnet152/weights/conv4_block25_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block25_3_bn.bin",
                           true,36,t1,t2);
    ConvBlk6 conv4_block26(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block26_1_conv.bin","../../../weights/resnet152/weights/conv4_block26_1_bn.bin","../../../weights/resnet152/weights/conv4_block26_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block26_2_bn.bin","../../../weights/resnet152/weights/conv4_block26_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block26_3_bn.bin",
                           true,37,t1,t2);
    ConvBlk6 conv4_block27(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block27_1_conv.bin","../../../weights/resnet152/weights/conv4_block27_1_bn.bin","../../../weights/resnet152/weights/conv4_block27_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block27_2_bn.bin","../../../weights/resnet152/weights/conv4_block27_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block27_3_bn.bin",
                           true,38,t1,t2);
    ConvBlk6 conv4_block28(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block28_1_conv.bin","../../../weights/resnet152/weights/conv4_block28_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block28_2_conv.bin","../../../weights/resnet152/weights/conv4_block28_2_bn.bin","../../../weights/resnet152/weights/conv4_block28_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block28_3_bn.bin",
                           true,39,t1,t2);
    ConvBlk6 conv4_block29(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block29_1_conv.bin","../../../weights/resnet152/weights/conv4_block29_1_bn.bin","../../../weights/resnet152/weights/conv4_block29_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block29_2_bn.bin","../../../weights/resnet152/weights/conv4_block29_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block29_3_bn.bin",
                           true,40,t1,t2);
    ConvBlk6 conv4_block30(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block30_1_conv.bin","../../../weights/resnet152/weights/conv4_block30_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block30_2_conv.bin","../../../weights/resnet152/weights/conv4_block30_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block30_3_conv.bin","../../../weights/resnet152/weights/conv4_block30_3_bn.bin",
                           true,41,t1,t2);
    ConvBlk6 conv4_block31(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block31_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block31_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block31_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block31_2_bn.bin","../../../weights/resnet152/weights/conv4_block31_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block31_3_bn.bin",
                           true,42,t1,t2);
    ConvBlk6 conv4_block32(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block32_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block32_1_bn.bin","../../../weights/resnet152/weights/conv4_block32_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block32_2_bn.bin","../../../weights/resnet152/weights/conv4_block32_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block32_3_bn.bin",
                           true,43,t1,t2);
    ConvBlk6 conv4_block33(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block33_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block33_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block33_2_conv.bin","../../../weights/resnet152/weights/conv4_block33_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block33_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block33_3_bn.bin",
                           true,44,t1,t2);
    ConvBlk6 conv4_block34(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block34_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block34_1_bn.bin","../../../weights/resnet152/weights/conv4_block34_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block34_2_bn.bin","../../../weights/resnet152/weights/conv4_block34_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block34_3_bn.bin",
                           true,45,t1,t2);
    ConvBlk6 conv4_block35(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block35_1_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block35_1_bn.bin","../../../weights/resnet152/weights/conv4_block35_2_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block35_2_bn.bin","../../../weights/resnet152/weights/conv4_block35_3_conv.bin",
                           "../../../weights/resnet152/weights/conv4_block35_3_bn.bin",
                           true,46,t1,t2);
    ConvBlk6 conv4_block36(1,1024,14,14,"../../../weights/resnet152/weights/conv4_block36_1_conv.bin","../../../weights/resnet152/weights/conv4_block36_1_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block36_2_conv.bin","../../../weights/resnet152/weights/conv4_block36_2_bn.bin",
                           "../../../weights/resnet152/weights/conv4_block36_3_conv.bin","../../../weights/resnet152/weights/conv4_block36_3_bn.bin",
                           true,47,t1,t2);
    ConvBlk7 conv5_block1(1,1024,14,14,"../../../weights/resnet152/weights/conv5_block1_0_conv.bin","../../../weights/resnet152/weights/conv5_block1_0_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block1_1_conv.bin","../../../weights/resnet152/weights/conv5_block1_1_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block1_2_conv.bin","../../../weights/resnet152/weights/conv5_block1_2_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block1_3_conv.bin",
                          "../../../weights/resnet152/weights/conv5_block1_3_bn.bin",
                          true,48,t1,t2);
    ConvBlk8 conv5_block2(1,2048,7,7,"../../../weights/resnet152/weights/conv5_block2_1_conv.bin",
                          "../../../weights/resnet152/weights/conv5_block2_1_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block2_2_conv.bin","../../../weights/resnet152/weights/conv5_block2_2_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block2_3_conv.bin","../../../weights/resnet152/weights/conv5_block2_3_bn.bin",
                          true,49,t1,t2);
    ConvBlk8 conv5_block3(1,2048,7,7,"../../../weights/resnet152/weights/conv5_block3_1_conv.bin","../../../weights/resnet152/weights/conv5_block3_1_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block3_2_conv.bin","../../../weights/resnet152/weights/conv5_block3_2_bn.bin",
                          "../../../weights/resnet152/weights/conv5_block3_3_conv.bin",
                          "../../../weights/resnet152/weights/conv5_block3_3_bn.bin",
                          true,50,t1,t2);

    Pool avg_pool;
    avg_pool.initialize(1,2048,7,7,0,7,7,CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,1);
    Conv predict;
    predict.initialize(1,2048,1,1,1000,0,1,1,1,"../../../weights/resnet152/weights/predictions.bin");

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaMemcpy(dInput, input, 224 * 224 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    float *output;
    output = conv1_conv.forward(dInput);
    output = conv1_bn.forward(output);
    output = conv1_relu.forward(output);
    output = conv1_max_pool.forward(output);
    output = conv2_block1.forward(output);
    output = conv2_block2.forward(output);
    output = conv2_block3.forward(output);
    output = conv3_block1.forward(output);
    output = conv3_block2.forward(output);
    output = conv3_block3.forward(output);
    output = conv3_block4.forward(output);
    output = conv3_block5.forward(output);
    output = conv3_block6.forward(output);
    output = conv3_block7.forward(output);
    output = conv3_block8.forward(output);
    output = conv4_block1.forward(output);
    output = conv4_block2.forward(output);
    output = conv4_block3.forward(output);
    output = conv4_block4.forward(output);
    output = conv4_block5.forward(output);
    output = conv4_block6.forward(output);
    output = conv4_block7.forward(output);
    output = conv4_block8.forward(output);
    output = conv4_block9.forward(output);
    output = conv4_block10.forward(output);
    output = conv4_block11.forward(output);
    output = conv4_block12.forward(output);
    output = conv4_block13.forward(output);
    output = conv4_block14.forward(output);
    output = conv4_block15.forward(output);
    output = conv4_block16.forward(output);
    output = conv4_block17.forward(output);
    output = conv4_block18.forward(output);
    output = conv4_block19.forward(output);
    output = conv4_block20.forward(output);
    output = conv4_block21.forward(output);
    output = conv4_block22.forward(output);
    output = conv4_block23.forward(output);
    output = conv4_block24.forward(output);
    output = conv4_block25.forward(output);
    output = conv4_block26.forward(output);
    output = conv4_block27.forward(output);
    output = conv4_block28.forward(output);
    output = conv4_block29.forward(output);
    output = conv4_block30.forward(output);
    output = conv4_block31.forward(output);
    output = conv4_block32.forward(output);
    output = conv4_block33.forward(output);
    output = conv4_block34.forward(output);
    output = conv4_block35.forward(output);
    output = conv4_block36.forward(output);
    output = conv5_block1.forward(output);
    output = conv5_block2.forward(output);
    output = conv5_block3.forward(output);
    output = avg_pool.forward(output);
    float total = 0.0f;
    for(int i=0;i<100;++i){
        t1[i] = 0.0f;
        t2[i] = 0.0f;
    }
    for(int i=0;i<100;++i) {
        load_input(imagePath + to_string(i) + ".bin", 3 * 224 * 224, input);
        cudaMemcpy(dInput,input,224*224*3*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(event_start);
        output = conv1_conv.forward(dInput);
        output = conv1_bn.forward(output);
        output = conv1_relu.forward(output);
        output = conv1_max_pool.forward(output);
        output = conv2_block1.forward(output);
        output = conv2_block2.forward(output);
        output = conv2_block3.forward(output);
        output = conv3_block1.forward(output);
        output = conv3_block2.forward(output);
        output = conv3_block3.forward(output);
        output = conv3_block4.forward(output);
        output = conv3_block5.forward(output);
        output = conv3_block6.forward(output);
        output = conv3_block7.forward(output);
        output = conv3_block8.forward(output);
        output = conv4_block1.forward(output);
        output = conv4_block2.forward(output);
        output = conv4_block3.forward(output);
        output = conv4_block4.forward(output);
        output = conv4_block5.forward(output);
        output = conv4_block6.forward(output);
        output = conv4_block7.forward(output);
        output = conv4_block8.forward(output);
        output = conv4_block9.forward(output);
        output = conv4_block10.forward(output);
        output = conv4_block11.forward(output);
        output = conv4_block12.forward(output);
        output = conv4_block13.forward(output);
        output = conv4_block14.forward(output);
        output = conv4_block15.forward(output);
        output = conv4_block16.forward(output);
        output = conv4_block17.forward(output);
        output = conv4_block18.forward(output);
        output = conv4_block19.forward(output);
        output = conv4_block20.forward(output);
        output = conv4_block21.forward(output);
        output = conv4_block22.forward(output);
        output = conv4_block23.forward(output);
        output = conv4_block24.forward(output);
        output = conv4_block25.forward(output);
        output = conv4_block26.forward(output);
        output = conv4_block27.forward(output);
        output = conv4_block28.forward(output);
        output = conv4_block29.forward(output);
        output = conv4_block30.forward(output);
        output = conv4_block31.forward(output);
        output = conv4_block32.forward(output);
        output = conv4_block33.forward(output);
        output = conv4_block34.forward(output);
        output = conv4_block35.forward(output);
        output = conv4_block36.forward(output);
        output = conv5_block1.forward(output);
        output = conv5_block2.forward(output);
        output = conv5_block3.forward(output);
        output = avg_pool.forward(output);
        output = predict.forward(output);
        cudaDeviceSynchronize();
        float cuda_time;
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&cuda_time, event_start, event_stop);
        total += cuda_time;
    }
    std::ofstream file_out;
    file_out.open("resnet152_layers.txt", std::ofstream::out);
    for(int i=0;i<50;++i){
        string out = to_string(i+1) + "," + to_string(t1[i]/100);
        file_out << out<<endl;
    }
    file_out.close();
    unsigned int outputSize = 1000;
    float *hOutput = (float *)malloc(outputSize*sizeof(float));
    cudaMemcpy(hOutput,output,outputSize*sizeof(float),cudaMemcpyDeviceToHost);
    vector<float> out_bin;
    for(int i=0;i<outputSize;++i){
        out_bin.push_back(hOutput[i]);
    }
    std::ofstream ofp("s.bin", std::ios::out | std::ios::binary);
    ofp.write(reinterpret_cast<const char*>(out_bin.data()), out_bin.size() * sizeof(float));
    return 0;
}
