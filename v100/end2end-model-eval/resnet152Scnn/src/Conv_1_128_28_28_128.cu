#include "../inc/scnnOps.h"
__device__ void conv_1_128_28_28_128_switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result){
    switch (switch_condition) {
        case 0:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(0-s)] += result;
                }
            }
            break;
        case 1:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(1-s)] += result;
                }
            }
            break;
        case 2:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(2-s)] += result;
                }
            }
            break;
        case 3:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(3-s)] += result;
                }
            }
            break;
        case 4:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(4-s)] += result;
                }
            }
            break;
        case 5:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(5-s)] += result;
                }
            }
            break;
        case 6:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(6-s)] += result;
                }
            }
            break;
        case 7:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(7-s)] += result;
                }
            }
            break;
        case 8:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*7+(8-s)] += result;
                }
            }
            break;
        case 9:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(0-s)] += result;
                }
            }
            break;
        case 10:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(1-s)] += result;
                }
            }
            break;
        case 11:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(2-s)] += result;
                }
            }
            break;
        case 12:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(3-s)] += result;
                }
            }
            break;
        case 13:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(4-s)] += result;
                }
            }
            break;
        case 14:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(5-s)] += result;
                }
            }
            break;
        case 15:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(6-s)] += result;
                }
            }
            break;
        case 16:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(7-s)] += result;
                }
            }
            break;
        case 17:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*7+(8-s)] += result;
                }
            }
            break;
        case 18:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(0-s)] += result;
                }
            }
            break;
        case 19:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(1-s)] += result;
                }
            }
            break;
        case 20:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(2-s)] += result;
                }
            }
            break;
        case 21:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(3-s)] += result;
                }
            }
            break;
        case 22:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(4-s)] += result;
                }
            }
            break;
        case 23:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(5-s)] += result;
                }
            }
            break;
        case 24:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(6-s)] += result;
                }
            }
            break;
        case 25:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(7-s)] += result;
                }
            }
            break;
        case 26:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*7+(8-s)] += result;
                }
            }
            break;
        case 27:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(0-s)] += result;
                }
            }
            break;
        case 28:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(1-s)] += result;
                }
            }
            break;
        case 29:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(2-s)] += result;
                }
            }
            break;
        case 30:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(3-s)] += result;
                }
            }
            break;
        case 31:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(4-s)] += result;
                }
            }
            break;
        case 32:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(5-s)] += result;
                }
            }
            break;
        case 33:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(6-s)] += result;
                }
            }
            break;
        case 34:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(7-s)] += result;
                }
            }
            break;
        case 35:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*7+(8-s)] += result;
                }
            }
            break;

    }
}
__global__ void conv_1_128_28_28_128_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens){
    unsigned int global_id = blockIdx.x * 512 + threadIdx.x;
    unsigned int batch_id = global_id/(128*28*28);
    if(global_id >= 1*28*28*128){
        return ;
    }
    const float v = matrix[global_id];
    if(v == 0.0f){
        return;
    }
    unsigned int c = global_id % 128;

    int h = ((global_id - batch_id * 128 * 28 * 28)/128)/28+1;
    int w = ((global_id - batch_id * 128 * 28 * 28)/128)%28+1;
    int th_start = min(h/2,14-1);
    int tw_start = min(w/7,4-1);
    for(int tile_h_id = th_start;tile_h_id>=0;tile_h_id--){
        if((tile_h_id*2+2+3-1)<=h){
            break;
        }
        for(int tile_w_id = tw_start;tile_w_id>=0;tile_w_id--){
            if((tile_w_id*7+7+3-1)<=w){
                break;
            }
            unsigned int tile_id = tile_h_id * 4 + tile_w_id;
            unsigned int index = atomicAdd(&c_lens[batch_id*128*56+c*56+tile_id],1);
            unsigned int abs_h = h - tile_h_id*2;
            unsigned int abs_w = w - tile_w_id*7;
            d_nnz[batch_id*128*56*(2+3-1)*(7+3-1)+
                  c*56*(2+3-1)*(7+3-1)+tile_id*(2+3-1)*(7+3-1)+index] = v;
            ids[batch_id*128*56*(2+3-1)*(7+3-1)+
                c*56*(2+3-1)*(7+3-1)+tile_id*(2+3-1)*(7+3-1)+index] = abs_h*(7+3-1)+abs_w;
        }
    }
}
__device__ void conv_1_128_28_28_128_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id){
    for(unsigned int r=0;r<3;++r){
        for(unsigned int s=0;s<3;++s){
            data_array[r*3+s] = kernel[c_index*128*3*3+r*3*128+s*128+n_id];
        }
    }
}
__device__ void conv_1_128_28_28_128_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                                float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                                unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                                unsigned int tile_id,unsigned int tile_c_id){
    for(unsigned int c_id=warp_id;c_id<8&&tile_c_id+c_id<128;c_id+=4){
        unsigned int end_index = c_lens[batch_id*128*56+(tile_c_id+c_id)*56+tile_id];
        if(lane_id ==0){
            shared_lens[c_id] = end_index;
        }
        for(unsigned int id = lane_id;id<end_index;id+=32){
            shared_input[c_id*(2+3-1)*(7+3-1)+id] = values[batch_id*128*56*(2+3-1)*(7+3-1)+
                                                           (tile_c_id+c_id)*56*(2+3-1)*(7+3-1)+tile_id*(2+3-1)*(7+3-1)+id];
            shared_ids[c_id*(2+3-1)*(7+3-1)+id] = ids[batch_id*128*56*(2+3-1)*(7+3-1)+
                                                      (tile_c_id+c_id)*56*(2+3-1)*(7+3-1)+tile_id*(2+3-1)*(7+3-1)+id];
        }
    }
}
__global__ void conv_1_128_28_28_128_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                            char * __restrict__ ids,
                                            const float * __restrict__ kernel, float * __restrict__ outputs){
    __shared__ float input[8*(2+3-1)*(7+3-1)];
    __shared__ char input_ids[8*(2+3-1)*(7+3-1)];
    __shared__ unsigned int channel_lens[(8)];

    const unsigned int batch_id = (blockIdx.x/(16*56));
    const unsigned int t_id = (blockIdx.x - batch_id*16*56)/16;
    const unsigned int tile_h_id = (t_id / 4)*2;
    const unsigned int tile_w_id = (t_id % 4)*7;
    const unsigned int index = blockIdx.x % (16);
    const unsigned int start_channel_index = index*8;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    float data_array[9];
    float temp_result[2*7] = {0.0f};
    conv_1_128_28_28_128_load_input_2_shared_memory(values,c_lens,ids,input,input_ids,channel_lens,warp_id,lane_id,batch_id,t_id,start_channel_index);
    __syncthreads();
    float v;
    unsigned int id;
    for(unsigned int n = threadIdx.x;n<128;n+=128){
        for(unsigned int c=start_channel_index;c<start_channel_index+8&&c<128;c++){
            unsigned int abs_c = c - start_channel_index;
            unsigned int start_index = abs_c*(2+3-1)*(7+3-1);
            unsigned int end_index = start_index+channel_lens[abs_c];
            if(start_index == end_index){
                continue;
            }
            conv_1_128_28_28_128_load_data_2_register(data_array,(c),kernel,n);
            unsigned int iters = end_index - start_index;
            for(unsigned int iter=0;iter<iters;iter++) {
                v = input[iter+start_index];
                id = input_ids[iter+start_index];
                conv_1_128_28_28_128_switch_function(id,data_array,v,temp_result);
            }
        }
        for (unsigned int th = 0; th < 2; ++th) {
            for (unsigned int tw = 0; tw < 7; ++tw) {
                if (tile_h_id + th >= 28 || tile_w_id + tw >= 28) {
                    continue;
                }
                atomicAdd(&outputs[batch_id * 128 * 28 * 28 + (tile_h_id + th) * 28 * 128 + (tile_w_id + tw) * 128 +
                                   n],temp_result[(th * 7 + tw)]);
            }
        }
        for(unsigned int i=0;i<2*7;++i){
            temp_result[i] = 0.0f;
        }
    }
}
void Conv_1_128_28_28_128::initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                                      unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile) {
    this->B = b;
    this->C = c;
    this->H = h;
    this->W = w;
    this->N = n;
    this->hOut = h;
    this->wOut = w;
    cudnnCreate(&convCudnn);
    unsigned int kernelSize = 3*3*C*N + N;//kernel + bias
    this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
    try{
        load_input(weightFile,kernelSize,cpuKernel);
    }catch (const char* msg) {
        cerr << msg << endl;
    }
    float *temp_kernel = (float *)malloc(kernelSize*sizeof(float));
    for(unsigned int i=0;i<N;++i){
        for(unsigned int l=0;l<C;++l){
            for(unsigned int j=0;j<3;++j){
                for(unsigned int k=0;k<3;++k){
                    temp_kernel[l*N*3*3+j*3*N+k*N+i] = cpuKernel[i*3*3*C+l*9+j*3+k];
                }
            }
        }
    }
    cudaMalloc(&kernel,9*C*N*sizeof(float));
    cudaMemcpy(kernel,temp_kernel,9*C*N*sizeof(float),cudaMemcpyHostToDevice);
    free(temp_kernel);
    cudaMalloc(&bias,N*sizeof(float));
    cudaMemcpy(bias,&cpuKernel[9*C*N],N*sizeof(float),cudaMemcpyHostToDevice);
    free(cpuKernel);
    TILES_EACH_CHANNEL = ((H-1)/TH+1)*((W-1)/TW+1);
    TCS = (C-1)/TC + 1;
    cudaMalloc(&nnz,b*c*TILES_EACH_CHANNEL*(TH+3-1)*(TW+3-1)*sizeof(float));
    cudaMalloc(&ids,b*c*TILES_EACH_CHANNEL*(TH+3-1)*(TW+3-1)*sizeof(char));
    cudaMalloc(&c_lens,b*c*TILES_EACH_CHANNEL*sizeof(unsigned int));
    cudaMalloc(&output,b*n*h*w*sizeof(float));
    cudnnCreateTensorDescriptor(&convOutputDescriptor);
    cudnnSetTensor4dDescriptor(convOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/N,
            /*image_height=*/hOut,
            /*image_width=*/wOut);
    cudnnCreateTensorDescriptor(&biasDescriptor);
    cudnnSetTensor4dDescriptor(biasDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/N,
            /*image_height=*/1,
            /*image_width=*/1);
}
float * Conv_1_128_28_28_128::forward(float *input) {
    cudaMemset(output, 0, B*N*hOut*wOut*sizeof(float));
    cudaMemset(c_lens, 0, B*C*TILES_EACH_CHANNEL*sizeof(unsigned int));
    conv_1_128_28_28_128_transform<<<(B*C*H*W-1)/512+1,512>>>(input,this->nnz,this->ids,this->c_lens);
    conv_1_128_28_28_128_conv2d<<<B*TCS*TILES_EACH_CHANNEL,N>>>(this->nnz,this->c_lens,this->ids,this->kernel,this->output);

    //chkerr(cudaGetLastError());
    //chkerr(cudaDeviceSynchronize());
    checkCUDNN(cudnnAddTensor(convCudnn,&alpha,biasDescriptor,bias,&beta2,convOutputDescriptor,output));
    return output;
}
