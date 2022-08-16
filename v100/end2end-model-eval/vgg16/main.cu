
#include "./inc/conv_and_pool.h"
#include "./inc/nvml_helper.hpp"
int main(int argc, char *argv[]){
    float *input = (float *)malloc(224*224*3*sizeof(float));
    string imagePath = argv[1];
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));

    Conv_and_pool block1(1,3,224,224,64,64,"../../../weights/vgg16/weights/block1_conv1.bin",
                         "../../../weights/vgg16/weights/block1_conv2.bin");
    Conv_and_pool block2(1,64,112,112,128,128,"../../../weights/vgg16/weights/block2_conv1.bin",
                         "../../../weights/vgg16/weights/block2_conv2.bin");
    Conv_and_pool3 block3(1,128,56,56,256,256,256,
                          "../../../weights/vgg16/weights/block3_conv1.bin",
                          "../../../weights/vgg16/weights/block3_conv2.bin",
                          "../../../weights/vgg16/weights/block3_conv3.bin");
    Conv_and_pool3 block4(1,256,28,28,512,512,512,
                          "../../../weights/vgg16/weights/block4_conv1.bin",
                          "../../../weights/vgg16/weights/block4_conv2.bin",
                          "../../../weights/vgg16/weights/block4_conv3.bin");
    Conv_and_pool3 block5(1,512,14,14,512,512,512,
                          "../../../weights/vgg16/weights/block5_conv1.bin",
                          "../../../weights/vgg16/weights/block5_conv2.bin",
                          "../../../weights/vgg16/weights/block5_conv3.bin");
    FC fc1;
    fc1.initialize(1,25088,1,1,4096,0,1,1,1,"../../../weights/vgg16/weights/fc1.bin");
    FC fc2;
    fc2.initialize(1,4096,1,1,4096,0,1,1,1,"../../../weights/vgg16/weights/fc2.bin");
    FC fc3;
    fc3.initialize(1,4096,1,1,1000,0,1,1,1,"../../../weights/vgg16/weights/predictions.bin");
    cudaMemcpy(dInput,input,224*224*3*sizeof(float),cudaMemcpyHostToDevice);
    float *output;
    output = block1.forward(dInput);
    output = block2.forward(output);
    output = block3.forward(output);
    output = block4.forward(output);
    output = block5.forward(output);
    output = fc1.forward(output);
    output = fc2.forward(output);
    output = fc3.forward(output);
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
        output = block1.forward(dInput);
        output = block2.forward(output);
        output = block3.forward(output);
        output = block4.forward(output);
        output = block5.forward(output);
        output = fc1.forward(output);
        output = fc2.forward(output);
        output = fc3.forward(output);
        cudaDeviceSynchronize();
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    cout<<"vgg16 cuDNN costs "<<inference_time/100<<" ms"<<endl;
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
