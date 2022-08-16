#include "../inc/convBlk.h"
#include "../inc/nvml_helper.hpp"

int main(int argc,char *argv[]){
    float *input = (float *)malloc(224*224*3*sizeof(float));
    string imagePath = argv[1];
    float *t1 = new float[100];
    float *t2 = new float[100];
    //load_input(imagePath,3*224*224,input);
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    Conv conv1_conv;
    conv1_conv.initialize(1,3,224,224,64,3,7,7,2,"../../../weights/densenet201/weights/conv1_conv.bin");
    BatchNorm conv1_bn;
    conv1_bn.initialize(1,64,112,112,"../../../weights/densenet201/weights/conv1_bn.bin");
    Activation conv1_relu;
    conv1_relu.initialize(1,64,112,112);
    Pool pool1;
    pool1.initialize(1,64,112,112,1,3,3,CUDNN_POOLING_MAX,2);
    ConvBlk56 conv2_block1(1,64,56,56,128,32,"../../../weights/densenet201/weights/conv2_block1_0_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block1_1_conv.bin",
                           "../../../weights/densenet201/weights/conv2_block1_1_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block1_2_conv.bin", true,1,t1,t2);
    ConvBlk56 conv2_block2(1,96,56,56,128,32,"../../../weights/densenet201/weights/conv2_block2_0_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block2_1_conv.bin","../../../weights/densenet201/weights/conv2_block2_1_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block2_2_conv.bin",true,2,t1,t2);
    ConvBlk56 conv2_block3(1,128,56,56,128,32,"../../../weights/densenet201/weights/conv2_block3_0_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block3_1_conv.bin",
                           "../../../weights/densenet201/weights/conv2_block3_1_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block3_2_conv.bin",true,3,t1,t2);
    ConvBlk56 conv2_block4(1,160,56,56,128,32,"../../../weights/densenet201/weights/conv2_block4_0_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block4_1_conv.bin","../../../weights/densenet201/weights/conv2_block4_1_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block4_2_conv.bin",true,4,t1,t2);
    ConvBlk56 conv2_block5(1,192,56,56,128,32,"../../../weights/densenet201/weights/conv2_block5_0_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block5_1_conv.bin",
                           "../../../weights/densenet201/weights/conv2_block5_1_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block5_2_conv.bin",true,5,t1,t2);
    ConvBlk56 conv2_block6(1,224,56,56,128,32,"../../../weights/densenet201/weights/conv2_block6_0_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block6_1_conv.bin",
                           "../../../weights/densenet201/weights/conv2_block6_1_bn.bin",
                           "../../../weights/densenet201/weights/conv2_block6_2_conv.bin",true,6,t1,t2);
    PoolBlk pool2(1,256,56,56,128,"../../../weights/densenet201/weights/pool2_bn.bin","../../../weights/densenet201/weights/pool2_conv.bin");
    ConvBlk28 conv3_block1(1,128,28,28,128,32,"../../../weights/densenet201/weights/conv3_block1_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block1_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block1_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block1_2_conv.bin",true,7,t1,t2);
    ConvBlk28 conv3_block2(1,160,28,28,128,32,"../../../weights/densenet201/weights/conv3_block2_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block2_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block2_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block2_2_conv.bin",true,8,t1,t2);
    ConvBlk28 conv3_block3(1,192,28,28,128,32,"../../../weights/densenet201/weights/conv3_block3_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block3_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block3_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block3_2_conv.bin",true,9,t1,t2);
    ConvBlk28 conv3_block4(1,224,28,28,128,32,"../../../weights/densenet201/weights/conv3_block4_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block4_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block4_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block4_2_conv.bin", true,10,t1,t2);
    ConvBlk28 conv3_block5(1,256,28,28,128,32,"../../../weights/densenet201/weights/conv3_block5_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block5_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block5_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block5_2_conv.bin", true,11,t1,t2);
    ConvBlk28 conv3_block6(1,288,28,28,128,32,"../../../weights/densenet201/weights/conv3_block6_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block6_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block6_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block6_2_conv.bin", true,12,t1,t2);
    ConvBlk28 conv3_block7(1,320,28,28,128,32,"../../../weights/densenet201/weights/conv3_block7_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block7_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block7_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block7_2_conv.bin",true,13,t1,t2);
    ConvBlk28 conv3_block8(1,352,28,28,128,32,"../../../weights/densenet201/weights/conv3_block8_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block8_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block8_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block8_2_conv.bin",true,14,t1,t2);
    ConvBlk28 conv3_block9(1,384,28,28,128,32,"../../../weights/densenet201/weights/conv3_block9_0_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block9_1_conv.bin",
                           "../../../weights/densenet201/weights/conv3_block9_1_bn.bin",
                           "../../../weights/densenet201/weights/conv3_block9_2_conv.bin",true,15,t1,t2);
    ConvBlk28 conv3_block10(1,416,28,28,128,32,"../../../weights/densenet201/weights/conv3_block10_0_bn.bin",
                            "../../../weights/densenet201/weights/conv3_block10_1_conv.bin",
                            "../../../weights/densenet201/weights/conv3_block10_1_bn.bin",
                            "../../../weights/densenet201/weights/conv3_block10_2_conv.bin",true,16,t1,t2);
    ConvBlk28 conv3_block11(1,448,28,28,128,32,"../../../weights/densenet201/weights/conv3_block11_0_bn.bin",
                            "../../../weights/densenet201/weights/conv3_block11_1_conv.bin",
                            "../../../weights/densenet201/weights/conv3_block11_1_bn.bin",
                            "../../../weights/densenet201/weights/conv3_block11_2_conv.bin",true,17,t1,t2);
    ConvBlk28 conv3_block12(1,480,28,28,128,32,"../../../weights/densenet201/weights/conv3_block12_0_bn.bin",
                            "../../../weights/densenet201/weights/conv3_block12_1_conv.bin",
                            "../../../weights/densenet201/weights/conv3_block12_1_bn.bin",
                            "../../../weights/densenet201/weights/conv3_block12_2_conv.bin",true,18,t1,t2);
    PoolBlk pool3(1,512,28,28,256,"../../../weights/densenet201/weights/pool3_bn.bin","../../../weights/densenet201/weights/pool3_conv.bin");
    ConvBlk14 conv4_block1(1,256,14,14,128,32,"../../../weights/densenet201/weights/conv4_block1_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block1_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block1_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block1_2_conv.bin", true,19,t1,t2);
    ConvBlk14 conv4_block2(1,288,14,14,128,32,"../../../weights/densenet201/weights/conv4_block2_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block2_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block2_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block2_2_conv.bin",true,20,t1,t2);
    ConvBlk14 conv4_block3(1,320,14,14,128,32,"../../../weights/densenet201/weights/conv4_block3_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block3_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block3_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block3_2_conv.bin",true,21,t1,t2);
    ConvBlk14 conv4_block4(1,352,14,14,128,32,"../../../weights/densenet201/weights/conv4_block4_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block4_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block4_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block4_2_conv.bin",true,22,t1,t2);
    ConvBlk14 conv4_block5(1,384,14,14,128,32,"../../../weights/densenet201/weights/conv4_block5_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block5_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block5_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block5_2_conv.bin",true,23,t1,t2);
    ConvBlk14 conv4_block6(1,416,14,14,128,32,"../../../weights/densenet201/weights/conv4_block6_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block6_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block6_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block6_2_conv.bin",true,24,t1,t2);
    ConvBlk14 conv4_block7(1,448,14,14,128,32,"../../../weights/densenet201/weights/conv4_block7_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block7_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block7_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block7_2_conv.bin", true,25,t1,t2);
    ConvBlk14 conv4_block8(1,480,14,14,128,32,"../../../weights/densenet201/weights/conv4_block8_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block8_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block8_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block8_2_conv.bin", true,26,t1,t2);
    ConvBlk14 conv4_block9(1,512,14,14,128,32,"../../../weights/densenet201/weights/conv4_block9_0_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block9_1_conv.bin",
                           "../../../weights/densenet201/weights/conv4_block9_1_bn.bin",
                           "../../../weights/densenet201/weights/conv4_block9_2_conv.bin", true,27,t1,t2);
    ConvBlk14 conv4_block10(1,544,14,14,128,32,"../../../weights/densenet201/weights/conv4_block10_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block10_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block10_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block10_2_conv.bin", true,28,t1,t2);
    ConvBlk14 conv4_block11(1,576,14,14,128,32,"../../../weights/densenet201/weights/conv4_block11_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block11_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block11_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block11_2_conv.bin",true,29,t1,t2);
    ConvBlk14 conv4_block12(1,608,14,14,128,32,"../../../weights/densenet201/weights/conv4_block12_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block12_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block12_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block12_2_conv.bin",true,30,t1,t2);
    ConvBlk14 conv4_block13(1,640,14,14,128,32,"../../../weights/densenet201/weights/conv4_block13_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block13_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block13_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block13_2_conv.bin",true,31,t1,t2);
    ConvBlk14 conv4_block14(1,672,14,14,128,32,"../../../weights/densenet201/weights/conv4_block14_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block14_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block14_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block14_2_conv.bin",true,32,t1,t2);
    ConvBlk14 conv4_block15(1,704,14,14,128,32,"../../../weights/densenet201/weights/conv4_block15_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block15_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block15_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block15_2_conv.bin",true,33,t1,t2);
    ConvBlk14 conv4_block16(1,736,14,14,128,32,"../../../weights/densenet201/weights/conv4_block16_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block16_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block16_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block16_2_conv.bin",true,34,t1,t2);
    ConvBlk14 conv4_block17(1,768,14,14,128,32,"../../../weights/densenet201/weights/conv4_block17_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block17_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block17_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block17_2_conv.bin",true,35,t1,t2);
    ConvBlk14 conv4_block18(1,800,14,14,128,32,"../../../weights/densenet201/weights/conv4_block18_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block18_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block18_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block18_2_conv.bin", true,36,t1,t2);
    ConvBlk14 conv4_block19(1,832,14,14,128,32,"../../../weights/densenet201/weights/conv4_block19_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block19_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block19_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block19_2_conv.bin",true,37,t1,t2);
    ConvBlk14 conv4_block20(1,864,14,14,128,32,"../../../weights/densenet201/weights/conv4_block20_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block20_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block20_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block20_2_conv.bin", true,38,t1,t2);
    ConvBlk14 conv4_block21(1,896,14,14,128,32,"../../../weights/densenet201/weights/conv4_block21_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block21_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block21_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block21_2_conv.bin", true,39,t1,t2);
    ConvBlk14 conv4_block22(1,928,14,14,128,32,"../../../weights/densenet201/weights/conv4_block22_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block22_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block22_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block22_2_conv.bin", true,40,t1,t2);
    ConvBlk14 conv4_block23(1,960,14,14,128,32,"../../../weights/densenet201/weights/conv4_block23_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block23_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block23_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block23_2_conv.bin",true,41,t1,t2);
    ConvBlk14 conv4_block24(1,992,14,14,128,32,"../../../weights/densenet201/weights/conv4_block24_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block24_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block24_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block24_2_conv.bin",true,42,t1,t2);
    ConvBlk14 conv4_block25(1,1024,14,14,128,32,"../../../weights/densenet201/weights/conv4_block25_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block25_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block25_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block25_2_conv.bin",true,43,t1,t2);
    ConvBlk14 conv4_block26(1,1056,14,14,128,32,"../../../weights/densenet201/weights/conv4_block26_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block26_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block26_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block26_2_conv.bin",true,44,t1,t2);
    ConvBlk14 conv4_block27(1,1088,14,14,128,32,"../../../weights/densenet201/weights/conv4_block27_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block27_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block27_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block27_2_conv.bin",true,45,t1,t2);
    ConvBlk14 conv4_block28(1,1120,14,14,128,32,"../../../weights/densenet201/weights/conv4_block28_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block28_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block28_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block28_2_conv.bin",true,46,t1,t2);
    ConvBlk14 conv4_block29(1,1152,14,14,128,32,"../../../weights/densenet201/weights/conv4_block29_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block29_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block29_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block29_2_conv.bin",true,47,t1,t2);
    ConvBlk14 conv4_block30(1,1184,14,14,128,32,"../../../weights/densenet201/weights/conv4_block30_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block30_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block30_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block30_2_conv.bin",true,48,t1,t2);
    ConvBlk14 conv4_block31(1,1216,14,14,128,32,"../../../weights/densenet201/weights/conv4_block31_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block31_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block31_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block31_2_conv.bin",true,49,t1,t2);
    ConvBlk14 conv4_block32(1,1248,14,14,128,32,"../../../weights/densenet201/weights/conv4_block32_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block32_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block32_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block32_2_conv.bin",true,50,t1,t2);
    ConvBlk14 conv4_block33(1,1280,14,14,128,32,"../../../weights/densenet201/weights/conv4_block33_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block33_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block33_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block33_2_conv.bin",true,51,t1,t2);
    ConvBlk14 conv4_block34(1,1312,14,14,128,32,"../../../weights/densenet201/weights/conv4_block34_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block34_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block34_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block34_2_conv.bin",true,52,t1,t2);
    ConvBlk14 conv4_block35(1,1344,14,14,128,32,"../../../weights/densenet201/weights/conv4_block35_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block35_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block35_1_bn.bin","../../../weights/densenet201/weights/conv4_block35_2_conv.bin",
                            true,53,t1,t2);
    ConvBlk14 conv4_block36(1,1376,14,14,128,32,"../../../weights/densenet201/weights/conv4_block36_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block36_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block36_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block36_2_conv.bin",true,54,t1,t2);
    ConvBlk14 conv4_block37(1,1408,14,14,128,32,"../../../weights/densenet201/weights/conv4_block37_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block37_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block37_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block37_2_conv.bin",true,55,t1,t2);
    ConvBlk14 conv4_block38(1,1440,14,14,128,32,"../../../weights/densenet201/weights/conv4_block38_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block38_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block38_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block38_2_conv.bin",true,56,t1,t2);
    ConvBlk14 conv4_block39(1,1472,14,14,128,32,"../../../weights/densenet201/weights/conv4_block39_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block39_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block39_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block39_2_conv.bin",true,57,t1,t2);
    ConvBlk14 conv4_block40(1,1504,14,14,128,32,"../../../weights/densenet201/weights/conv4_block40_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block40_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block40_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block40_2_conv.bin",true,58,t1,t2);
    ConvBlk14 conv4_block41(1,1536,14,14,128,32,"../../../weights/densenet201/weights/conv4_block41_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block41_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block41_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block41_2_conv.bin", true,59,t1,t2);
    ConvBlk14 conv4_block42(1,1568,14,14,128,32,"../../../weights/densenet201/weights/conv4_block42_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block42_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block42_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block42_2_conv.bin", true,60,t1,t2);
    ConvBlk14 conv4_block43(1,1600,14,14,128,32,"../../../weights/densenet201/weights/conv4_block43_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block43_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block43_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block43_2_conv.bin", true,61,t1,t2);
    ConvBlk14 conv4_block44(1,1632,14,14,128,32,"../../../weights/densenet201/weights/conv4_block44_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block44_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block44_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block44_2_conv.bin",true,62,t1,t2);
    ConvBlk14 conv4_block45(1,1664,14,14,128,32,"../../../weights/densenet201/weights/conv4_block45_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block45_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block45_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block45_2_conv.bin", true,63,t1,t2);
    ConvBlk14 conv4_block46(1,1696,14,14,128,32,"../../../weights/densenet201/weights/conv4_block46_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block46_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block46_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block46_2_conv.bin", true,64,t1,t2);
    ConvBlk14 conv4_block47(1,1728,14,14,128,32,"../../../weights/densenet201/weights/conv4_block47_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block47_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block47_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block47_2_conv.bin", true,65,t1,t2);
    ConvBlk14 conv4_block48(1,1760,14,14,128,32,"../../../weights/densenet201/weights/conv4_block48_0_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block48_1_conv.bin",
                            "../../../weights/densenet201/weights/conv4_block48_1_bn.bin",
                            "../../../weights/densenet201/weights/conv4_block48_2_conv.bin",true,66,t1,t2);
    PoolBlk pool4(1,1792,14,14,896,"../../../weights/densenet201/weights/pool4_bn.bin","../../../weights/densenet201/weights/pool4_conv.bin");
    ConvBlk7 conv5_block1(1,896,7,7,128,32,"../../../weights/densenet201/weights/conv5_block1_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block1_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block1_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block1_2_conv.bin",true,67,t1,t2);
    ConvBlk7 conv5_block2(1,928,7,7,128,32,"../../../weights/densenet201/weights/conv5_block2_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block2_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block2_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block2_2_conv.bin", true,68,t1,t2);
    ConvBlk7 conv5_block3(1,960,7,7,128,32,"../../../weights/densenet201/weights/conv5_block3_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block3_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block3_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block3_2_conv.bin", true,69,t1,t2);
    ConvBlk7 conv5_block4(1,992,7,7,128,32,"../../../weights/densenet201/weights/conv5_block4_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block4_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block4_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block4_2_conv.bin",true,70,t1,t2);
    ConvBlk7 conv5_block5(1,1024,7,7,128,32,"../../../weights/densenet201/weights/conv5_block5_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block5_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block5_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block5_2_conv.bin", true,71,t1,t2);
    ConvBlk7 conv5_block6(1,1056,7,7,128,32,"../../../weights/densenet201/weights/conv5_block6_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block6_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block6_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block6_2_conv.bin",true,72,t1,t2);
    ConvBlk7 conv5_block7(1,1088,7,7,128,32,"../../../weights/densenet201/weights/conv5_block7_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block7_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block7_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block7_2_conv.bin",true,73,t1,t2);
    ConvBlk7 conv5_block8(1,1120,7,7,128,32,"../../../weights/densenet201/weights/conv5_block8_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block8_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block8_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block8_2_conv.bin",true,74,t1,t2);
    ConvBlk7 conv5_block9(1,1152,7,7,128,32,"../../../weights/densenet201/weights/conv5_block9_0_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block9_1_conv.bin",
                          "../../../weights/densenet201/weights/conv5_block9_1_bn.bin",
                          "../../../weights/densenet201/weights/conv5_block9_2_conv.bin",true,75,t1,t2);
    ConvBlk7 conv5_block10(1,1184,7,7,128,32,"../../../weights/densenet201/weights/conv5_block10_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block10_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block10_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block10_2_conv.bin",true,76,t1,t2);
    ConvBlk7 conv5_block11(1,1216,7,7,128,32,"../../../weights/densenet201/weights/conv5_block11_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block11_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block11_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block11_2_conv.bin",true,77,t1,t2);
    ConvBlk7 conv5_block12(1,1248,7,7,128,32,"../../../weights/densenet201/weights/conv5_block12_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block12_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block12_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block12_2_conv.bin",true,78,t1,t2);
    ConvBlk7 conv5_block13(1,1280,7,7,128,32,"../../../weights/densenet201/weights/conv5_block13_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block13_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block13_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block13_2_conv.bin",true,79,t1,t2);
    ConvBlk7 conv5_block14(1,1312,7,7,128,32,"../../../weights/densenet201/weights/conv5_block14_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block14_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block14_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block14_2_conv.bin",true,80,t1,t2);
    ConvBlk7 conv5_block15(1,1344,7,7,128,32,"../../../weights/densenet201/weights/conv5_block15_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block15_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block15_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block15_2_conv.bin",true,81,t1,t2);
    ConvBlk7 conv5_block16(1,1376,7,7,128,32,"../../../weights/densenet201/weights/conv5_block16_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block16_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block16_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block16_2_conv.bin",true,82,t1,t2);
    ConvBlk7 conv5_block17(1,1408,7,7,128,32,"../../../weights/densenet201/weights/conv5_block17_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block17_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block17_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block17_2_conv.bin",true,83,t1,t2);
    ConvBlk7 conv5_block18(1,1440,7,7,128,32,"../../../weights/densenet201/weights/conv5_block18_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block18_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block18_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block18_2_conv.bin",true,84,t1,t2);
    ConvBlk7 conv5_block19(1,1472,7,7,128,32,"../../../weights/densenet201/weights/conv5_block19_0_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block19_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block19_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block19_2_conv.bin",true,85,t1,t2);
    ConvBlk7 conv5_block20(1,1504,7,7,128,32,"../../../weights/densenet201/weights/conv5_block20_0_bn.bin","../../../weights/densenet201/weights/conv5_block20_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block20_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block20_2_conv.bin",true,86,t1,t2);
    ConvBlk7 conv5_block21(1,1536,7,7,128,32,"../../../weights/densenet201/weights/conv5_block21_0_bn.bin","../../../weights/densenet201/weights/conv5_block21_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block21_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block21_2_conv.bin",true,87,t1,t2);
    ConvBlk7 conv5_block22(1,1568,7,7,128,32,"../../../weights/densenet201/weights/conv5_block22_0_bn.bin","../../../weights/densenet201/weights/conv5_block22_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block22_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block22_2_conv.bin",true,88,t1,t2);
    ConvBlk7 conv5_block23(1,1600,7,7,128,32,"../../../weights/densenet201/weights/conv5_block23_0_bn.bin","../../../weights/densenet201/weights/conv5_block23_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block23_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block23_2_conv.bin",true,89,t1,t2);
    ConvBlk7 conv5_block24(1,1632,7,7,128,32,"../../../weights/densenet201/weights/conv5_block24_0_bn.bin","../../../weights/densenet201/weights/conv5_block24_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block24_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block24_2_conv.bin",true,90,t1,t2);
    ConvBlk7 conv5_block25(1,1664,7,7,128,32,"../../../weights/densenet201/weights/conv5_block25_0_bn.bin","../../../weights/densenet201/weights/conv5_block25_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block25_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block25_2_conv.bin",true,91,t1,t2);
    ConvBlk7 conv5_block26(1,1696,7,7,128,32,"../../../weights/densenet201/weights/conv5_block26_0_bn.bin","../../../weights/densenet201/weights/conv5_block26_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block26_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block26_2_conv.bin",true,92,t1,t2);
    ConvBlk7 conv5_block27(1,1728,7,7,128,32,"../../../weights/densenet201/weights/conv5_block27_0_bn.bin","../../../weights/densenet201/weights/conv5_block27_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block27_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block27_2_conv.bin",true,93,t1,t2);
    ConvBlk7 conv5_block28(1,1760,7,7,128,32,"../../../weights/densenet201/weights/conv5_block28_0_bn.bin","../../../weights/densenet201/weights/conv5_block28_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block28_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block28_2_conv.bin",true,94,t1,t2);
    ConvBlk7 conv5_block29(1,1792,7,7,128,32,"../../../weights/densenet201/weights/conv5_block29_0_bn.bin","../../../weights/densenet201/weights/conv5_block29_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block29_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block29_2_conv.bin",true,95,t1,t2);
    ConvBlk7 conv5_block30(1,1824,7,7,128,32,"../../../weights/densenet201/weights/conv5_block30_0_bn.bin","../../../weights/densenet201/weights/conv5_block30_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block30_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block30_2_conv.bin",true,96,t1,t2);
    ConvBlk7 conv5_block31(1,1856,7,7,128,32,"../../../weights/densenet201/weights/conv5_block31_0_bn.bin","../../../weights/densenet201/weights/conv5_block31_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block31_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block31_2_conv.bin",true,97,t1,t2);
    ConvBlk7 conv5_block32(1,1888,7,7,128,32,"../../../weights/densenet201/weights/conv5_block32_0_bn.bin","../../../weights/densenet201/weights/conv5_block32_1_conv.bin",
                           "../../../weights/densenet201/weights/conv5_block32_1_bn.bin",
                           "../../../weights/densenet201/weights/conv5_block32_2_conv.bin",true,98,t1,t2);
    BatchNorm bn;
    bn.initialize(1,1920,7,7,"../../../weights/densenet201/weights/bn.bin");
    Activation relu;
    relu.initialize(1,1920,7,7);
    Pool avg_pool;
    avg_pool.initialize(1,1920,7,7,0,7,7,CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,1);
    FC predict;
    predict.initialize(1,1920,1,1,1000,0,1,1,1,"../../../weights/densenet201/weights/predictions.bin");
    cudaMemcpy(dInput,input,224*224*3*sizeof(float),cudaMemcpyHostToDevice);
    float *output;
    output = conv1_conv.forward(dInput);
    output = conv1_bn.forward(output);
    output = conv1_relu.forward(output);
    output = pool1.forward(output);
    output = conv2_block1.forward(output);
    output = conv2_block2.forward(output);
    output = conv2_block3.forward(output);
    output = conv2_block4.forward(output);
    output = conv2_block5.forward(output);
    output = conv2_block6.forward(output);
    output = pool2.forward(output);
    output = conv3_block1.forward(output);
    output = conv3_block2.forward(output);
    output = conv3_block3.forward(output);
    output = conv3_block4.forward(output);
    output = conv3_block5.forward(output);
    output = conv3_block6.forward(output);
    output = conv3_block7.forward(output);
    output = conv3_block8.forward(output);
    output = conv3_block9.forward(output);
    output = conv3_block10.forward(output);
    output = conv3_block11.forward(output);
    output = conv3_block12.forward(output);
    output = pool3.forward(output);
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
    output = conv4_block37.forward(output);
    output = conv4_block38.forward(output);
    output = conv4_block39.forward(output);
    output = conv4_block40.forward(output);
    output = conv4_block41.forward(output);
    output = conv4_block42.forward(output);
    output = conv4_block43.forward(output);
    output = conv4_block44.forward(output);
    output = conv4_block45.forward(output);
    output = conv4_block46.forward(output);
    output = conv4_block47.forward(output);
    output = conv4_block48.forward(output);
    output = pool4.forward(output);
    output = conv5_block1.forward(output);
    output = conv5_block2.forward(output);
    output = conv5_block3.forward(output);
    output = conv5_block4.forward(output);
    output = conv5_block5.forward(output);
    output = conv5_block6.forward(output);
    output = conv5_block7.forward(output);
    output = conv5_block8.forward(output);
    output = conv5_block9.forward(output);
    output = conv5_block10.forward(output);
    output = conv5_block11.forward(output);
    output = conv5_block12.forward(output);
    output = conv5_block13.forward(output);
    output = conv5_block14.forward(output);
    output = conv5_block15.forward(output);
    output = conv5_block16.forward(output);
    output = conv5_block17.forward(output);
    output = conv5_block18.forward(output);
    output = conv5_block19.forward(output);
    output = conv5_block20.forward(output);
    output = conv5_block21.forward(output);
    output = conv5_block22.forward(output);
    output = conv5_block23.forward(output);
    output = conv5_block24.forward(output);
    output = conv5_block25.forward(output);
    output = conv5_block26.forward(output);
    output = conv5_block27.forward(output);
    output = conv5_block28.forward(output);
    output = conv5_block29.forward(output);
    output = conv5_block30.forward(output);
    output = conv5_block31.forward(output);
    output = conv5_block32.forward(output);
    output = bn.forward(output);
    output = relu.forward(output);
    output = avg_pool.forward(output);
    output = predict.forward(output);
    for(int i=0;i<100;++i){
        t1[i] = 0.0f;
        t2[i] = 0.0f;
    }
    float total = 0.0f;
    for(int i=0;i<100;++i){
        load_input(imagePath + to_string(i) + ".bin", 3 * 224 * 224, input);
        cudaMemcpy(dInput,input,224*224*3*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(event_start);
        output = conv1_conv.forward(dInput);
        output = conv1_bn.forward(output);
        output = conv1_relu.forward(output);
        output = pool1.forward(output);
        output = conv2_block1.forward(output);
        output = conv2_block2.forward(output);
        output = conv2_block3.forward(output);
        output = conv2_block4.forward(output);
        output = conv2_block5.forward(output);
        output = conv2_block6.forward(output);
        output = pool2.forward(output);
        output = conv3_block1.forward(output);
        output = conv3_block2.forward(output);
        output = conv3_block3.forward(output);
        output = conv3_block4.forward(output);
        output = conv3_block5.forward(output);
        output = conv3_block6.forward(output);
        output = conv3_block7.forward(output);
        output = conv3_block8.forward(output);
        output = conv3_block9.forward(output);
        output = conv3_block10.forward(output);
        output = conv3_block11.forward(output);
        output = conv3_block12.forward(output);
        output = pool3.forward(output);
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
        output = conv4_block37.forward(output);
        output = conv4_block38.forward(output);
        output = conv4_block39.forward(output);
        output = conv4_block40.forward(output);
        output = conv4_block41.forward(output);
        output = conv4_block42.forward(output);
        output = conv4_block43.forward(output);
        output = conv4_block44.forward(output);
        output = conv4_block45.forward(output);
        output = conv4_block46.forward(output);
        output = conv4_block47.forward(output);
        output = conv4_block48.forward(output);
        output = pool4.forward(output);
        output = conv5_block1.forward(output);
        output = conv5_block2.forward(output);
        output = conv5_block3.forward(output);
        output = conv5_block4.forward(output);
        output = conv5_block5.forward(output);
        output = conv5_block6.forward(output);
        output = conv5_block7.forward(output);
        output = conv5_block8.forward(output);
        output = conv5_block9.forward(output);
        output = conv5_block10.forward(output);
        output = conv5_block11.forward(output);
        output = conv5_block12.forward(output);
        output = conv5_block13.forward(output);
        output = conv5_block14.forward(output);
        output = conv5_block15.forward(output);
        output = conv5_block16.forward(output);
        output = conv5_block17.forward(output);
        output = conv5_block18.forward(output);
        output = conv5_block19.forward(output);
        output = conv5_block20.forward(output);
        output = conv5_block21.forward(output);
        output = conv5_block22.forward(output);
        output = conv5_block23.forward(output);
        output = conv5_block24.forward(output);
        output = conv5_block25.forward(output);
        output = conv5_block26.forward(output);
        output = conv5_block27.forward(output);
        output = conv5_block28.forward(output);
        output = conv5_block29.forward(output);
        output = conv5_block30.forward(output);
        output = conv5_block31.forward(output);
        output = conv5_block32.forward(output);
        output = bn.forward(output);
        output = relu.forward(output);
        output = avg_pool.forward(output);
        output = predict.forward(output);
        cudaDeviceSynchronize();
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float cuda_time;
        cudaEventElapsedTime(&cuda_time, event_start, event_stop);
        total += cuda_time;
    }
    std::ofstream file_out;
    file_out.open ("densenet201_layers.txt", std::ofstream::out);
    for(int i=0;i<99;++i){
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