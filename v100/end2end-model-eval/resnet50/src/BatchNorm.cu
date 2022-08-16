#include "../inc/common.h"
void BatchNorm::initialize(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string weight) {
    B = b;
    H = h;
    W = w;
    C = c;
    chkerr(cudaMalloc(&scaleDev,C*sizeof(float)));
    chkerr(cudaMalloc(&shiftDev,C*sizeof(float)));
    chkerr(cudaMalloc(&meanDev,C*sizeof(float)));
    chkerr(cudaMalloc(&varDev,C*sizeof(float)));
    checkCUDNN(cudnnCreate(&batchNormCudnn));
    cudaMalloc(&output,B*C*H*W*sizeof(float));
    checkCUDNN(cudnnCreateTensorDescriptor(&batchNormInputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(batchNormInputDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&batchNormOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(batchNormOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/C,
            /*image_height=*/1,
            /*image_width=*/1));

    this->cpuKernel = (float *)malloc(4*C*sizeof(float));
    //load_input(weight,4*C,cpuKernel);
    try{
        load_input(weight,4*C,cpuKernel);
    }catch (const char* msg) {
        cerr << msg << endl;
    }
    chkerr(cudaMemcpy(scaleDev,cpuKernel,C*sizeof(float),cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(shiftDev,&cpuKernel[C],C*sizeof(float),cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(meanDev,&cpuKernel[2*C],C*sizeof(float),cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(varDev,&cpuKernel[3*C],C*sizeof(float),cudaMemcpyHostToDevice));
    free(cpuKernel);
}
float * BatchNorm::forward(float *input) {
    checkCUDNN(cudnnBatchNormalizationForwardInference(
            batchNormCudnn,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            batchNormInputDescriptor,
            input, //gpu上的
            batchNormOutputDescriptor,
            output, //gpu上的
            bnScaleBiasMeanVarDesc,
            scaleDev,  //gpu上的
            shiftDev,    //gpu上的
            meanDev,  //gpu上的
            varDev,//gpu上的
            CUDNN_BN_MIN_EPSILON
    ));
    return output;
}
/*int main(void){
    BatchNorm batchNorm;
    batchNorm.initialize(1,112,112,64,
                        "../weights/conv1_bn_0.bin","../weights/conv1_bn_1.bin",
                        "../weights/conv1_bn_2.bin","../weights/conv1_bn_3.bin");
    float *input;
    float *hostInput = (float *)malloc((1*64*112*112)*sizeof(float));
    for(int i=0;i<1*64*112*112;++i){
        hostInput[i] = 1.0f;
    }
    cudaMalloc(&input,1*64*112*112*sizeof(float));
    cudaMemcpy(input,hostInput,1*64*112*112*sizeof(float),cudaMemcpyHostToDevice);

    //conv.forward(input);
    //float *outputPython = load_input("../conv.bin",1*112*112*64);
    float *outputCudnn = (float *)malloc(1*112*112*64*sizeof(float));
    cudaMemcpy(outputCudnn,batchNorm.forward(input),1*112*112*64*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<outputCudnn[63]<<endl;
    float diff = 0.0f;
    for(int i=0;i<112*112*64;i++){
        diff +=(outputCudnn[i] - outputPython[i]);
    }
    cout<<outputCudnn[63]<<" "<<outputPython[0]<<endl;
    cout<<diff<<endl;
    return 0;
}*/