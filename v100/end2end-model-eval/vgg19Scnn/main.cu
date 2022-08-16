#include "./inc/conv_and_pool.h"
#include "./inc/nvml_helper.hpp"

int main(int argc, char *argv[]){
    float *input = (float *)malloc(224*224*3*sizeof(float));
    string imagePath = argv[1];
    load_input(imagePath+"/" + to_string(0) + ".bin",3*224*224,input);
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));

    Conv block1_conv1;
    block1_conv1.initialize(1,3,224,224,64,1,3,3,1,
                            "../../../weights/vgg19/weights/block1_conv1.bin");
    Relu relu1;
    relu1.initialize(1,64,224,224,0.62);
    Conv_1_64_224_224_64 block1_conv2;
    block1_conv2.initialize(1,64,224,224,64,1,3,3,
                            1,"../../../weights/vgg19/weights/block1_conv2.bin");
    Conv block1_conv2_cudnn;
    block1_conv2_cudnn.initialize(1,64,224,224,64,1,3,3,1,
                                  "../../../weights/vgg19/weights/block1_conv2.bin");
    Activation relu2;
    relu2.initialize(1,64,224,224);

    Pool block1_pool;
    block1_pool.initialize(1,64,224,224,1,3,3,CUDNN_POOLING_MAX,2);
    Conv block2_conv1;
    block2_conv1.initialize(1,64,112,112,128,1,3,3,1,
                            "../../../weights/vgg19/weights/block2_conv1.bin");
    Relu relu3;
    relu3.initialize(1,128,112,112,0.75);
    Conv_1_128_112_112_128 block2_conv2;
    Conv block2_conv2_cudnn;
    block2_conv2.initialize(1,128,112,112,128,1,3,3,1,
                            "../../../weights/vgg19/weights/block2_conv2.bin");
    block2_conv2_cudnn.initialize(1,128,112,112,128,1,3,3,1,
                                  "../../../weights/vgg19/weights/block2_conv2.bin");
    Activation relu4;
    relu4.initialize(1,128,112,112);

    Pool block2_pool;
    block2_pool.initialize(1,128,112,112,1,3,3,CUDNN_POOLING_MAX,2);
    Conv block3_conv1;
    block3_conv1.initialize(1,128,56,56,256,1,3,3,1,
                            "../../../weights/vgg19/weights/block3_conv1.bin");
    Relu relu5;
    relu5.initialize(1,256,56,56,0.6);
    Conv_1_256_56_56_256 block3_conv2;
    Conv block3_conv2_cudnn;
    block3_conv2.initialize(1,256,56,56,256,1,3,3,1,
                            "../../../weights/vgg19/weights/block3_conv2.bin");
    block3_conv2_cudnn.initialize(1,256,56,56,256,1,3,3,
                                  1,"../../../weights/vgg19/weights/block3_conv2.bin");
    Relu relu6;
    relu6.initialize(1,256,56,56,0.6);
    Conv_1_256_56_56_256 block3_conv3;
    Conv block3_conv3_cudnn;
    block3_conv3.initialize(1,256,56,56,256,1,3,3,
                            1,"../../../weights/vgg19/weights/block3_conv3.bin");
    block3_conv3_cudnn.initialize(1,256,56,56,256,1,3,3,
                                  1,"../../../weights/vgg19/weights/block3_conv3.bin");
    Relu relu7;
    relu7.initialize(1,256,56,56,0.6);
    Conv block3_conv4_cudnn;
    Conv_1_256_56_56_256 block3_conv4;
    block3_conv4.initialize(1,256,56,56,256,1,3,3,
                            1,"../../../weights/vgg19/weights/block3_conv4.bin");
    block3_conv4_cudnn.initialize(1,256,56,56,256,1,3,3,
                                  1,"../../../weights/vgg19/weights/block3_conv4.bin");
    Activation relu8;
    relu8.initialize(1,256,56,56);

    Pool block3_pool;
    block3_pool.initialize(1,256,56,56,1,3,3,CUDNN_POOLING_MAX,2);
    Conv block4_conv1;
    block4_conv1.initialize(1,256,28,28,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block4_conv1.bin");
    Relu relu9;
    relu9.initialize(1,512,28,28,0.4);
    Conv block4_conv2_cudnn;
    Conv_1_512_28_28_512 block4_conv2;
    block4_conv2.initialize(1,512,28,28,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block4_conv2.bin");
    block4_conv2_cudnn.initialize(1,512,28,28,512,1,3,3,
                                  1,"../../../weights/vgg19/weights/block4_conv2.bin");
    Relu relu10;
    relu10.initialize(1,512,28,28,0.4);
    Conv block4_conv3_cudnn;
    Conv_1_512_28_28_512 block4_conv3;
    block4_conv3.initialize(1,512,28,28,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block4_conv3.bin");
    block4_conv3_cudnn.initialize(1,512,28,28,512,1,3,3,
                                  1,"../../../weights/vgg19/weights/block4_conv3.bin");
    Relu relu11;
    relu11.initialize(1,512,28,28,0.4);
    Conv block4_conv4_cudnn;
    Conv_1_512_28_28_512 block4_conv4;
    block4_conv4.initialize(1,512,28,28,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block4_conv4.bin");
    block4_conv4_cudnn.initialize(1,512,28,28,512,1,3,3,
                                  1,"../../../weights/vgg19/weights/block4_conv4.bin");
    Activation relu12;
    relu12.initialize(1,512,28,28);


    Pool block4_pool;
    block4_pool.initialize(1,512,28,28,1,3,3,CUDNN_POOLING_MAX,2);
    Conv block5_conv1;
    block5_conv1.initialize(1,512,14,14,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block5_conv1.bin");
    Relu relu13;
    relu13.initialize(1,512,14,14,0.4);
    Conv block5_conv2_cudnn;
    Conv_1_512_14_14_512 block5_conv2;
    block5_conv2.initialize(1,512,14,14,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block5_conv2.bin");
    block5_conv2_cudnn.initialize(1,512,14,14,512,1,3,3,
                                  1,"../../../weights/vgg19/weights/block5_conv2.bin");
    Activation relu14;
    relu14.initialize(1,512,14,14);
    Conv block5_conv3_cudnn;
    Conv_1_512_14_14_512 block5_conv3;
    block5_conv3.initialize(1,512,14,14,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block5_conv3.bin");
    block5_conv3_cudnn.initialize(1,512,14,14,512,1,3,3,
                                  1,"../../../weights/vgg19/weights/block5_conv3.bin");
    Activation relu15;
    relu15.initialize(1,512,14,14);
    Conv block5_conv4_cudnn;
    Conv_1_512_14_14_512 block5_conv4;
    block5_conv4.initialize(1,512,14,14,512,1,3,3,
                            1,"../../../weights/vgg19/weights/block5_conv4.bin");
    block5_conv4_cudnn.initialize(1,512,14,14,512,1,3,3,
                                  1,"../../../weights/vgg19/weights/block5_conv4.bin");
    Activation relu16;
    relu16.initialize(1,512,14,14);

    Pool block5_pool;
    block5_pool.initialize(1,512,14,14,1,3,3,CUDNN_POOLING_MAX,2);
    FC fc1;
    fc1.initialize(1,25088,1,1,4096,0,1,1,1,"../../../weights/vgg19/weights/fc1.bin");
    FC fc2;
    fc2.initialize(1,4096,1,1,4096,0,1,1,1,"../../../weights/vgg19/weights/fc2.bin");
    FC fc3;
    fc3.initialize(1,4096,1,1,1000,0,1,1,1,"../../../weights/vgg19/weights/predictions.bin");

    cudaMemcpy(dInput,input,224*224*3*sizeof(float),cudaMemcpyHostToDevice);
    float *output;
    output = block1_conv1.forward(dInput);
    output = relu1.forward(output);
    if(relu1.sparse){
        output = block1_conv2.forward(output);
    }else{
        output = block1_conv2_cudnn.forward(output);
    }
    output = relu2.forward(output);
    output = block1_pool.forward(output);
    output = block2_conv1.forward(output);
    output = relu3.forward(output);
    if(relu3.sparse){
        output = block2_conv2.forward(output);
    }else{
        output = block2_conv2_cudnn.forward(output);
    }
    output = relu4.forward(output);
    output = block2_pool.forward(output);
    output = block3_conv1.forward(output);
    output = relu5.forward(output);
    if(relu5.sparse){
        output = block3_conv2.forward(output);
    }else{
        output = block3_conv2_cudnn.forward(output);
    }
    output = relu6.forward(output);
    if(relu6.sparse){
        output = block3_conv3.forward(output);
    }else{
        output = block3_conv3_cudnn.forward(output);
    }
    output = relu7.forward(output);
    if(relu7.sparse){
        output = block3_conv4.forward(output);
    }else{
        output = block3_conv4_cudnn.forward(output);
    }
    output = relu8.forward(output);
    output = block3_pool.forward(output);
    output = block4_conv1.forward(output);
    output = relu9.forward(output);
    if(relu9.sparse){
        output = block4_conv2.forward(output);
    }else{
        output = block4_conv2_cudnn.forward(output);
    }
    output = relu10.forward(output);
    if(relu10.sparse){
        output = block4_conv3.forward(output);
    }else{
        output = block4_conv3_cudnn.forward(output);
    }
    output = relu11.forward(output);
    if(relu11.sparse){
        output = block4_conv4.forward(output);
    }else{
        output = block4_conv4_cudnn.forward(output);
    }
    output = relu12.forward(output);
    output = block4_pool.forward(output);
    output = block5_conv1.forward(output);
    output = relu13.forward(output);
    if(relu13.sparse){
        output = block5_conv2.forward(output);
    }else{
        output = block5_conv2_cudnn.forward(output);
    }
    output = relu14.forward(output);
    output = block5_conv3.forward(output);
    output = relu15.forward(output);
    output = block5_conv4.forward(output);
    output = relu16.forward(output);
    output = block5_pool.forward(output);
    output = fc1.forward(output);
    output = fc2.forward(output);
    output = fc3.forward(output);
    cudaDeviceSynchronize();
    float inference_time = 0.0f;
    for(int i=0;i<100;++i){
        string image = imagePath + "/" + to_string(i) + ".bin";
        load_input(image, 3 * 224 * 224, input);
        cudaMemcpy(dInput,input,224*224*3*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(event_start);
        output = block1_conv1.forward(dInput);
        output = relu1.forward(output);
        if(relu1.sparse){
            output = block1_conv2.forward(output);
        }else{
            output = block1_conv2_cudnn.forward(output);
        }
        output = relu2.forward(output);
        output = block1_pool.forward(output);
        output = block2_conv1.forward(output);
        output = relu3.forward(output);
        if(relu3.sparse){
            output = block2_conv2.forward(output);
        }else{
            output = block2_conv2_cudnn.forward(output);
        }
        output = relu4.forward(output);
        output = block2_pool.forward(output);
        output = block3_conv1.forward(output);
        output = relu5.forward(output);
        if(relu5.sparse){
            output = block3_conv2.forward(output);
        }else{
            output = block3_conv2_cudnn.forward(output);
        }
        output = relu6.forward(output);
        if(relu6.sparse){
            output = block3_conv3.forward(output);
        }else{
            output = block3_conv3_cudnn.forward(output);
        }
        output = relu7.forward(output);
        if(relu7.sparse){
            output = block3_conv4.forward(output);
        }else{
            output = block3_conv4_cudnn.forward(output);
        }
        output = relu8.forward(output);
        output = block3_pool.forward(output);
        output = block4_conv1.forward(output);
        output = relu9.forward(output);
        if(relu9.sparse){
            output = block4_conv2.forward(output);
        }else{
            output = block4_conv2_cudnn.forward(output);
        }
        output = relu10.forward(output);
        if(relu10.sparse){
            output = block4_conv3.forward(output);
        }else{
            output = block4_conv3_cudnn.forward(output);
        }
        output = relu11.forward(output);
        if(relu11.sparse){
            output = block4_conv4.forward(output);
        }else{
            output = block4_conv4_cudnn.forward(output);
        }
        output = relu12.forward(output);
        output = block4_pool.forward(output);
        output = block5_conv1.forward(output);
        output = relu13.forward(output);
        if(relu13.sparse){
            output = block5_conv2.forward(output);
        }else{
            output = block5_conv2_cudnn.forward(output);
        }
        output = relu14.forward(output);
        output = block5_conv3.forward(output);
        output = relu15.forward(output);
        output = block5_conv4.forward(output);
        output = relu16.forward(output);
        output = block5_pool.forward(output);
        output = fc1.forward(output);
        output = fc2.forward(output);
        output = fc3.forward(output);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    cout<<"vgg19 cuSNN costs "<<inference_time/100<<" ms"<<endl;
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
