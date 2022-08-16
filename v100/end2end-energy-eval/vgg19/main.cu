
#include "./inc/conv_and_pool.h"
#include "./inc/nvml_helper.hpp"
void generate_random_input(float * array, unsigned int size){
    for(unsigned int i=0;i<size;++i){
        array[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
    }
}
int main(int argc, char *argv[]){
    int test_images = 100;
    float *input = (float *)malloc(224*224*3*sizeof(float)*test_images);
    string imagePath = argv[1];
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float)*test_images);

    Conv_and_pool block1(1,3,224,224,64,64,"../../../weights/vgg19/weights/block1_conv1.bin","../../../weights/vgg19/weights/block1_conv2.bin");
    Conv_and_pool block2(1,64,112,112,128,128,"../../../weights/vgg19/weights/block2_conv1.bin","../../../weights/vgg19/weights/block2_conv2.bin");
    Conv_and_pool4 block3(1,128,56,56,256,256,256,256,"../../../weights/vgg19/weights/block3_conv1.bin",
                          "../../../weights/vgg19/weights/block3_conv2.bin","../../../weights/vgg19/weights/block3_conv3.bin","../../../weights/vgg19/weights/block3_conv4.bin");
    Conv_and_pool4 block4(1,256,28,28,512,512,512,512,"../../../weights/vgg19/weights/block4_conv1.bin",
                          "../../../weights/vgg19/weights/block4_conv2.bin","../../../weights/vgg19/weights/block4_conv3.bin","../../../weights/vgg19/weights/block4_conv4.bin");
    Conv_and_pool4 block5(1,512,14,14,512,512,512,512,"../../../weights/vgg19/weights/block5_conv1.bin",
                          "../../../weights/vgg19/weights/block5_conv2.bin","../../../weights/vgg19/weights/block5_conv3.bin","../../../weights/vgg19/weights/block5_conv4.bin");
    FC fc1;
    fc1.initialize(1,25088,1,1,4096,0,1,1,1,"../../../weights/vgg19/weights/fc1.bin");
    FC fc2;
    fc2.initialize(1,4096,1,1,4096,0,1,1,1,"../../../weights/vgg19/weights/fc2.bin");
    FC fc3;
    fc3.initialize(1,4096,1,1,1000,0,1,1,1,"../../../weights/vgg19/weights/predictions.bin");

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

    generate_random_input(input, test_images*3*224*224);
    cudaMemcpy(dInput,input,test_images*3*224*224*sizeof(float),cudaMemcpyHostToDevice);
    NVML_MEASURE(dev0,&start);
    for(int i=0;i<test_images;++i){
        output = block1.forward(dInput);
        output = block2.forward(output);
        output = block3.forward(output);
        output = block4.forward(output);
        output = block5.forward(output);
        output = fc1.forward(output);
        output = fc2.forward(output);
        output = fc3.forward(output);
        cudaDeviceSynchronize();
    }

    NVML_MEASURE(dev0,&end);
    energy +=(end - start);
    cout<<"vgg19 cuDNN energy costs,"<<energy/test_images<<" mj"<<endl;
    cout<<endl;
    return 0;
}
