#include "../inc/common.h"
#include "../inc/scnn.h"
__global__ void relu(float * __restrict__ values, unsigned int B,unsigned int C,unsigned int H,
                     unsigned int W,unsigned int *counter){
    __shared__ unsigned int block_count[1];
    block_count[0] = 0;
    __syncthreads();
    unsigned int local_count = 0;
    unsigned int lane_id = threadIdx.x % 32;
    for(unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;i<B*C*H*W;i+=gridDim.x*blockDim.x){
        float value = values[i];
        if(value<=0.0f){
            values[i] = 0.0f;
            local_count ++;
        }
    }
    for(int j=16;j>0;j=j/2){
        local_count += __shfl_down_sync(0xFFFFFFFF,local_count,j);
    }
    if(lane_id == 0&&local_count>0){
        atomicAdd(&block_count[0],local_count);
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(&counter[0],block_count[0]);
    }
}