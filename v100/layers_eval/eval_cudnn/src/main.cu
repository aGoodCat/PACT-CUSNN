#include "../inc/cudnnOps.h"
float rand_FloatRange(float a, float b)
{
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}
float compute_difference(float *output1, float *output2, unsigned int size){
    float diff = 0;
    #pragma omp parallel for reduction (+:diff)
    for(int i=0;i<size;++i){
        diff += abs(output1[i] - output2[i]);
    }
    return diff;
}
float compute_sparsity(float *input, unsigned int size){
    float diff = 0;
    #pragma omp parallel for reduction (+:diff)
    for(int i=0;i<size;++i){
        if(input[i] == 0.0f){
            diff +=1;
        }
    }
    diff = diff/size;
    return diff;
}
void generate_random_input(unsigned int array_size, float * array){
    #pragma omp parallel for
    for(unsigned int i=0; i<array_size; ++i){
        array[i] = rand_FloatRange(1,100);
    }
}
float get_minimum(float a, float b, float c){
    float answer = min(a,b);
    answer = min(answer,c);
    return answer;
}
int main(int argc, char *argv[]){
    unsigned int B = atoi(argv[1]);
    unsigned int C = atoi(argv[2]);
    unsigned int H = atoi(argv[3]);
    unsigned int W = atoi(argv[4]);
    unsigned int N = atoi(argv[5]);
    string network = argv[6];
    string id = argv[7];
    unsigned int inputSize = B*H*W*C;
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM_CONV conv1;
    conv1.initialize(B,C,H,W,N,1,3,3,1);
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM_CONV conv2;
    conv2.initialize(B,C,H,W,N,1,3,3,1);
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM_CONV conv3;
    conv3.initialize(B,C,H,W,N,1,3,3,1);

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    float *input = new float[20*H*W*C];
    generate_random_input(20*H*W*C, input);
    float *dInput;
    cudaMalloc(&dInput,inputSize*sizeof(float));
    float times[4] = {0.0f};
    float inferenceTime;
    cudaMemcpy(dInput,input,B*C*H*W*sizeof(float),cudaMemcpyHostToDevice);
    float *output;
    float *output_cudnn;
    output_cudnn = conv1.forward(dInput);
    output_cudnn = conv2.forward(dInput);
    output_cudnn = conv3.forward(dInput);
    for(int i=0;i<20;i+=B){
        cudaMemcpy(dInput,&input[i*C*H*W],inputSize*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(event_start);
        output_cudnn = conv1.forward(dInput);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&inferenceTime, event_start, event_stop);
        times[0] += inferenceTime;

        cudaEventRecord(event_start);
        output_cudnn = conv2.forward(dInput);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&inferenceTime, event_start, event_stop);
        times[1] += inferenceTime;

        cudaEventRecord(event_start);
        output_cudnn = conv3.forward(dInput);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&inferenceTime, event_start, event_stop);
        times[2] += inferenceTime;
    }
    std::ofstream file_out;
    file_out.open ("cudnn_layers.txt", std::ofstream::out|std::ofstream::app);
    file_out<<network<<","<<id<<","<<get_minimum((times[0]*B)/20,(times[1]*B)/20,(times[2]*B)/20)<<endl;
    file_out.close();
    return 0;
}
