#include "../inc/common.h"
#include "../inc/scnn.h"
__device__ void conv_1_128_7_7_32_switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result){
    switch (switch_condition) {
        case 0:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*4+(0-s)] += result;
                }
            }
            break;
        case 1:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*4+(1-s)] += result;
                }
            }
            break;
        case 2:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*4+(2-s)] += result;
                }
            }
            break;
        case 3:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*4+(3-s)] += result;
                }
            }
            break;
        case 4:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*4+(4-s)] += result;
                }
            }
            break;
        case 5:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(0-r)*4+(5-s)] += result;
                }
            }
            break;
        case 6:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*4+(0-s)] += result;
                }
            }
            break;
        case 7:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*4+(1-s)] += result;
                }
            }
            break;
        case 8:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*4+(2-s)] += result;
                }
            }
            break;
        case 9:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*4+(3-s)] += result;
                }
            }
            break;
        case 10:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*4+(4-s)] += result;
                }
            }
            break;
        case 11:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(1-r)*4+(5-s)] += result;
                }
            }
            break;
        case 12:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*4+(0-s)] += result;
                }
            }
            break;
        case 13:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*4+(1-s)] += result;
                }
            }
            break;
        case 14:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*4+(2-s)] += result;
                }
            }
            break;
        case 15:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*4+(3-s)] += result;
                }
            }
            break;
        case 16:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*4+(4-s)] += result;
                }
            }
            break;
        case 17:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(2-r)*4+(5-s)] += result;
                }
            }
            break;
        case 18:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*4+(0-s)] += result;
                }
            }
            break;
        case 19:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*4+(1-s)] += result;
                }
            }
            break;
        case 20:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*4+(2-s)] += result;
                }
            }
            break;
        case 21:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*4+(3-s)] += result;
                }
            }
            break;
        case 22:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*4+(4-s)] += result;
                }
            }
            break;
        case 23:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*3+s];
                    temp_result[(3-r)*4+(5-s)] += result;
                }
            }
            break;

    }

}
__global__ void conv_1_128_7_7_32_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens){
    unsigned int global_id = blockIdx.x * 512 + threadIdx.x;
    unsigned int batch_id = global_id/(128*7*7);
    if(global_id >= 1*7*7*128){
        return ;
    }
    const float v = matrix[global_id];
    if(v == 0.0f){
        return;
    }
    unsigned int c = global_id % 128;

    int h = ((global_id - batch_id * 128 * 7 * 7)/128)/7+1;
    int w = ((global_id - batch_id * 128 * 7 * 7)/128)%7+1;
    int th_start = min(h/2,4-1);
    int tw_start = min(w/4,2-1);
    for(int tile_h_id = th_start;tile_h_id>=0;tile_h_id--){
        if((tile_h_id*2+2+3-1)<=h){
            break;
        }
        for(int tile_w_id = tw_start;tile_w_id>=0;tile_w_id--){
            if((tile_w_id*4+4+3-1)<=w){
                break;
            }
            unsigned int tile_id = tile_h_id * 2 + tile_w_id;
            unsigned int index = atomicAdd(&c_lens[batch_id*128*8+c*8+tile_id],1);
            unsigned int abs_h = h - tile_h_id*2;
            unsigned int abs_w = w - tile_w_id*4;
            d_nnz[batch_id*128*8*(2+3-1)*(4+3-1)+
                  c*8*(2+3-1)*(4+3-1)+tile_id*(2+3-1)*(4+3-1)+index] = v;
            ids[batch_id*128*8*(2+3-1)*(4+3-1)+
                c*8*(2+3-1)*(4+3-1)+tile_id*(2+3-1)*(4+3-1)+index] = abs_h*(4+3-1)+abs_w;
        }
    }

}
__device__ void conv_1_128_7_7_32_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id){
    for(unsigned int r=0;r<3;++r){
        for(unsigned int s=0;s<3;++s){
            data_array[r*3+s] = kernel[c_index*32*3*3+r*3*32+s*32+n_id];
        }
    }

}
__device__ void conv_1_128_7_7_32_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                             float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                             unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                             unsigned int tile_id,unsigned int tile_c_id){
    for(unsigned int c_id=warp_id;c_id<2&&tile_c_id+c_id<128;c_id+=1){
        unsigned int end_index = c_lens[batch_id*128*8+(tile_c_id+c_id)*8+tile_id];
        if(lane_id ==0){
            shared_lens[c_id] = end_index;
        }
        for(unsigned int id = lane_id;id<end_index;id+=32){
            shared_input[c_id*(2+3-1)*(4+3-1)+id] = values[batch_id*128*8*(2+3-1)*(4+3-1)+
                                                           (tile_c_id+c_id)*8*(2+3-1)*(4+3-1)+tile_id*(2+3-1)*(4+3-1)+id];
            shared_ids[c_id*(2+3-1)*(4+3-1)+id] = ids[batch_id*128*8*(2+3-1)*(4+3-1)+
                                                      (tile_c_id+c_id)*8*(2+3-1)*(4+3-1)+tile_id*(2+3-1)*(4+3-1)+id];
        }
    }

}
__global__ void conv_1_128_7_7_32_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                         char * __restrict__ ids,
                                         const float * __restrict__ kernel, float * __restrict__ outputs){
    __shared__ float input[2*(2+3-1)*(4+3-1)];
    __shared__ char input_ids[2*(2+3-1)*(4+3-1)];
    __shared__ unsigned int channel_lens[(2)];

    const unsigned int batch_id = (blockIdx.x/(64*8));
    const unsigned int t_id = (blockIdx.x - batch_id*64*8)/64;
    const unsigned int tile_h_id = (t_id / 2)*2;
    const unsigned int tile_w_id = (t_id % 2)*4;
    const unsigned int index = blockIdx.x % (64);
    const unsigned int start_channel_index = index*2;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    float data_array[9];
    float temp_result[2*4] = {0.0f};
    conv_1_128_7_7_32_load_input_2_shared_memory(values,c_lens,ids,input,input_ids,channel_lens,warp_id,lane_id,batch_id,t_id,start_channel_index);
    __syncthreads();
    float v;
    unsigned int id;
    for(unsigned int n = threadIdx.x;n<32;n+=32){
        for(unsigned int c=start_channel_index;c<start_channel_index+2&&c<128;c++){
            unsigned int abs_c = c - start_channel_index;
            unsigned int start_index = abs_c*(2+3-1)*(4+3-1);
            unsigned int end_index = start_index+channel_lens[abs_c];
            if(start_index == end_index){
                continue;
            }
            conv_1_128_7_7_32_load_data_2_register(data_array,(c),kernel,n);
            unsigned int iters = end_index - start_index;
            for(unsigned int iter=0;iter<iters;iter++) {
                v = input[iter+start_index];
                id = input_ids[iter+start_index];
                conv_1_128_7_7_32_switch_function(id,data_array,v,temp_result);
            }
        }
        for (unsigned int th = 0; th < 2; ++th) {
            for (unsigned int tw = 0; tw < 4; ++tw) {
                if (tile_h_id + th >= 7 || tile_w_id + tw >= 7) {
                    continue;
                }
                atomicAdd(&outputs[batch_id * 32 * 7 * 7 + (tile_h_id + th) * 7 * 32 + (tile_w_id + tw) * 32 +
                                   n],temp_result[(th * 4 + tw)]);
            }
        }
        for(unsigned int i=0;i<2*4;++i){
            temp_result[i] = 0.0f;
        }
    }
}
