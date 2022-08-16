#include "../inc/common.h"
#include "../inc/nvml_helper.hpp"
int main(int argc,char *argv[]){
    float *input = (float *)malloc(224*224*3*sizeof(float));
    string imagePath = argv[1];
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    Conv conv1_conv;
    conv1_conv.initialize(1,3,224,224,64,3,7,7,2,"../../../weights/resnet50/weights/conv1_conv.bin");
    BatchNorm conv1_bn;
    conv1_bn.initialize(1,64,112,112,"../../../weights/resnet50/weights/conv1_bn.bin");
    Activation conv1_relu;
    conv1_relu.initialize(1,64,112,112);
    Pool conv1_max_pool;
    conv1_max_pool.initialize(1,64,112,112,1,3,3,CUDNN_POOLING_MAX,2);

    ConvBlk1 conv2_block1(1,64,56,56,"../../../weights/resnet50/weights/conv2_block1_0_conv.bin","../../../weights/resnet50/weights/conv2_block1_0_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block1_1_conv.bin","../../../weights/resnet50/weights/conv2_block1_1_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block1_2_conv.bin","../../../weights/resnet50/weights/conv2_block1_2_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block1_3_conv.bin","../../../weights/resnet50/weights/conv2_block1_3_bn.bin");
    ConvBlk2 conv2_block2(1,256,56,56,"../../../weights/resnet50/weights/conv2_block2_1_conv.bin","../../../weights/resnet50/weights/conv2_block2_1_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block2_2_conv.bin","../../../weights/resnet50/weights/conv2_block2_2_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block2_3_conv.bin","../../../weights/resnet50/weights/conv2_block2_3_bn.bin");
    ConvBlk2 conv2_block3(1,256,56,56,"../../../weights/resnet50/weights/conv2_block3_1_conv.bin","../../../weights/resnet50/weights/conv2_block3_1_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block3_2_conv.bin","../../../weights/resnet50/weights/conv2_block3_2_bn.bin",
                          "../../../weights/resnet50/weights/conv2_block3_3_conv.bin","../../../weights/resnet50/weights/conv2_block3_3_bn.bin");

    ConvBlk3 conv3_block1(1,256,56,56,"../../../weights/resnet50/weights/conv3_block1_0_conv.bin","../../../weights/resnet50/weights/conv3_block1_0_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block1_1_conv.bin","../../../weights/resnet50/weights/conv3_block1_1_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block1_2_conv.bin","../../../weights/resnet50/weights/conv3_block1_2_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block1_3_conv.bin","../../../weights/resnet50/weights/conv3_block1_3_bn.bin");
    ConvBlk4 conv3_block2(1,512,28,28,"../../../weights/resnet50/weights/conv3_block2_1_conv.bin","../../../weights/resnet50/weights/conv3_block2_1_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block2_2_conv.bin","../../../weights/resnet50/weights/conv3_block2_2_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block2_3_conv.bin","../../../weights/resnet50/weights/conv3_block2_3_bn.bin");
    ConvBlk4 conv3_block3(1,512,28,28,"../../../weights/resnet50/weights/conv3_block3_1_conv.bin","../../../weights/resnet50/weights/conv3_block3_1_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block3_2_conv.bin","../../../weights/resnet50/weights/conv3_block3_2_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block3_3_conv.bin","../../../weights/resnet50/weights/conv3_block3_3_bn.bin");
    ConvBlk4 conv3_block4(1,512,28,28,"../../../weights/resnet50/weights/conv3_block4_1_conv.bin","../../../weights/resnet50/weights/conv3_block4_1_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block4_2_conv.bin","../../../weights/resnet50/weights/conv3_block4_2_bn.bin",
                          "../../../weights/resnet50/weights/conv3_block4_3_conv.bin","../../../weights/resnet50/weights/conv3_block4_3_bn.bin");

    ConvBlk5 conv4_block1(1,512,28,28,"../../../weights/resnet50/weights/conv4_block1_0_conv.bin","../../../weights/resnet50/weights/conv4_block1_0_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block1_1_conv.bin","../../../weights/resnet50/weights/conv4_block1_1_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block1_2_conv.bin","../../../weights/resnet50/weights/conv4_block1_2_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block1_3_conv.bin","../../../weights/resnet50/weights/conv4_block1_3_bn.bin");

    ConvBlk6 conv4_block2(1,1024,14,14,"../../../weights/resnet50/weights/conv4_block2_1_conv.bin","../../../weights/resnet50/weights/conv4_block2_1_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block2_2_conv.bin","../../../weights/resnet50/weights/conv4_block2_2_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block2_3_conv.bin","../../../weights/resnet50/weights/conv4_block2_3_bn.bin");
    ConvBlk6 conv4_block3(1,1024,14,14,"../../../weights/resnet50/weights/conv4_block3_1_conv.bin","../../../weights/resnet50/weights/conv4_block3_1_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block3_2_conv.bin","../../../weights/resnet50/weights/conv4_block3_2_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block3_3_conv.bin","../../../weights/resnet50/weights/conv4_block3_3_bn.bin");
    ConvBlk6 conv4_block4(1,1024,14,14,"../../../weights/resnet50/weights/conv4_block4_1_conv.bin","../../../weights/resnet50/weights/conv4_block4_1_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block4_2_conv.bin","../../../weights/resnet50/weights/conv4_block4_2_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block4_3_conv.bin","../../../weights/resnet50/weights/conv4_block4_3_bn.bin");
    ConvBlk6 conv4_block5(1,1024,14,14,"../../../weights/resnet50/weights/conv4_block5_1_conv.bin","../../../weights/resnet50/weights/conv4_block5_1_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block5_2_conv.bin","../../../weights/resnet50/weights/conv4_block5_2_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block5_3_conv.bin","../../../weights/resnet50/weights/conv4_block5_3_bn.bin");
    ConvBlk6 conv4_block6(1,1024,14,14,"../../../weights/resnet50/weights/conv4_block6_1_conv.bin","../../../weights/resnet50/weights/conv4_block6_1_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block6_2_conv.bin","../../../weights/resnet50/weights/conv4_block6_2_bn.bin",
                          "../../../weights/resnet50/weights/conv4_block6_3_conv.bin","../../../weights/resnet50/weights/conv4_block6_3_bn.bin");

    ConvBlk7 conv5_block1(1,1024,14,14,"../../../weights/resnet50/weights/conv5_block1_0_conv.bin","../../../weights/resnet50/weights/conv5_block1_0_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block1_1_conv.bin","../../../weights/resnet50/weights/conv5_block1_1_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block1_2_conv.bin","../../../weights/resnet50/weights/conv5_block1_2_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block1_3_conv.bin",
                          "../../../weights/resnet50/weights/conv5_block1_3_bn.bin");
    ConvBlk8 conv5_block2(1,2048,7,7,"../../../weights/resnet50/weights/conv5_block2_1_conv.bin",
                          "../../../weights/resnet50/weights/conv5_block2_1_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block2_2_conv.bin","../../../weights/resnet50/weights/conv5_block2_2_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block2_3_conv.bin","../../../weights/resnet50/weights/conv5_block2_3_bn.bin");
    ConvBlk8 conv5_block3(1,2048,7,7,"../../../weights/resnet50/weights/conv5_block3_1_conv.bin","../../../weights/resnet50/weights/conv5_block3_1_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block3_2_conv.bin","../../../weights/resnet50/weights/conv5_block3_2_bn.bin",
                          "../../../weights/resnet50/weights/conv5_block3_3_conv.bin",
                          "../../../weights/resnet50/weights/conv5_block3_3_bn.bin");
    Pool avg_pool;
    avg_pool.initialize(1,2048,7,7,0,7,7,CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,1);
    Conv predict;
    predict.initialize(1,2048,1,1,1000,0,1,1,1,"../../../weights/resnet50/weights/predictions.bin");

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
    output = conv4_block1.forward(output);
    output = conv4_block2.forward(output);
    output = conv4_block3.forward(output);
    output = conv4_block4.forward(output);
    output = conv4_block5.forward(output);
    output = conv4_block6.forward(output);
    output = conv5_block1.forward(output);
    output = conv5_block2.forward(output);
    output = conv5_block3.forward(output);
    output = avg_pool.forward(output);
    output = predict.forward(output);
    cudaDeviceSynchronize();
    NVML_INIT;
    NVML_DEV_t dev0;
    NVML_GET_HANDLE(0, &dev0);
    unsigned long long start,end;
    unsigned long long energy = 0;
    float inference_time = 0.0f;
    for(int i=0;i<100;++i){
        string image = imagePath + "/" + to_string(i) + ".bin";
        load_input(image, 3 * 224 * 224, input);
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
        output = conv4_block1.forward(output);
        output = conv4_block2.forward(output);
        output = conv4_block3.forward(output);
        output = conv4_block4.forward(output);
        output = conv4_block5.forward(output);
        output = conv4_block6.forward(output);
        output = conv5_block1.forward(output);
        output = conv5_block2.forward(output);
        output = conv5_block3.forward(output);
        output = avg_pool.forward(output);
        output = predict.forward(output);
        cudaDeviceSynchronize();
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    cout<<"resnet50 cuDNN costs "<<inference_time/100<<" ms"<<endl;
    cout<<endl;
    unsigned int outputSize = 1000;
    float *hOutput = (float *)malloc(outputSize*sizeof(float));
    cudaMemcpy(hOutput,output,outputSize*sizeof(float),cudaMemcpyDeviceToHost);
    vector<float> out_bin;
    for(int i=0;i<outputSize;++i){
        out_bin.push_back(hOutput[i]);
    }
    std::ofstream ofp("d.bin", std::ios::out | std::ios::binary);
    ofp.write(reinterpret_cast<const char*>(out_bin.data()), out_bin.size() * sizeof(float));
    return 0;
}
