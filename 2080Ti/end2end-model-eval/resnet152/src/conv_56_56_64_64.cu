class Conv56_64_64{
public:
    __global__ void transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens);
    __global__ void conv2d()
};
#define SC 64
#define SN 64
#define SH 56
#define SW 56
#define R 3
#define S 3
#define BLK_DIM 64
#define TC 4
#define TCS ((SC-1)/TC+1)
#define WARPS_CONV ((BLK_DIM-1)/32+1)
#define TH 4
#define TW 4
#define TILES_EACH_CHANNEL (((SHo-1)/TH+1)*((SWo-1)/TW+1))

__global__ void transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens){
    unsigned int global_id = blockIdx.x * BLK_DIM_TRANS + threadIdx.x;
    unsigned int batch_id = global_id/(SC*SH*SW);
    if(global_id >= SB*SH*SW*SC){
        return ;
    }
    const float v = matrix[global_id];
    if(v == 0.0f){
        return;
    }
    unsigned int c = global_id % SC;

    int h = ((global_id - batch_id * SC * SH * SW)/SC)/SW+1;
    int w = ((global_id - batch_id * SC * SH * SW)/SC)%SW+1;
    int th_start = min(h/TH,THS-1);
    int tw_start = min(w/TW,TWS-1);
    for(int tile_h_id = th_start;tile_h_id>=0;tile_h_id--){
        if((tile_h_id*TH+TH+R-1)<=h){
            break;
        }
        for(int tile_w_id = tw_start;tile_w_id>=0;tile_w_id--){
            if((tile_w_id*TW+TW+S-1)<=w){
                break;
            }
            unsigned int tile_id = tile_h_id * TWS + tile_w_id;
            unsigned int index = atomicAdd(&c_lens[batch_id*SC*TILES_EACH_CHANNEL+c*TILES_EACH_CHANNEL+tile_id],1);
            unsigned int abs_h = h - tile_h_id*TH;
            unsigned int abs_w = w - tile_w_id*TW;
            d_nnz[batch_id*SC*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+
                  c*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+tile_id*(TH+R-1)*(TW+S-1)+index] = v;
            ids[batch_id*SC*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+
                c*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+tile_id*(TH+R-1)*(TW+S-1)+index] = abs_h*(TW+S-1)+abs_w;
        }
    }
}
__device__ void load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id){
    for(unsigned int r=0;r<R;++r){
        for(unsigned int s=0;s<S;++s){
            data_array[r*S+s] = kernel[c_index*SN*R*S+r*S*SN+s*SN+n_id];
        }
    }
}
__device__ void switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result){
    switch (switch_condition) {
        case 0:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(0-r)*4+(0-s)] += result;
                }
            }
            break;
        case 1:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(0-r)*4+(1-s)] += result;
                }
            }
            break;
        case 2:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(0-r)*4+(2-s)] += result;
                }
            }
            break;
        case 3:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(0-r)*4+(3-s)] += result;
                }
            }
            break;
        case 4:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(0-r)*4+(4-s)] += result;
                }
            }
            break;
        case 5:
            for ( int r = 0; r < 1; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(0-r)*4+(5-s)] += result;
                }
            }
            break;
        case 6:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(1-r)*4+(0-s)] += result;
                }
            }
            break;
        case 7:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(1-r)*4+(1-s)] += result;
                }
            }
            break;
        case 8:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(1-r)*4+(2-s)] += result;
                }
            }
            break;
        case 9:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(1-r)*4+(3-s)] += result;
                }
            }
            break;
        case 10:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(1-r)*4+(4-s)] += result;
                }
            }
            break;
        case 11:
            for ( int r = 0; r < 2; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(1-r)*4+(5-s)] += result;
                }
            }
            break;
        case 12:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(2-r)*4+(0-s)] += result;
                }
            }
            break;
        case 13:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(2-r)*4+(1-s)] += result;
                }
            }
            break;
        case 14:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(2-r)*4+(2-s)] += result;
                }
            }
            break;
        case 15:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(2-r)*4+(3-s)] += result;
                }
            }
            break;
        case 16:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(2-r)*4+(4-s)] += result;
                }
            }
            break;
        case 17:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(2-r)*4+(5-s)] += result;
                }
            }
            break;
        case 18:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(3-r)*4+(0-s)] += result;
                }
            }
            break;
        case 19:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(3-r)*4+(1-s)] += result;
                }
            }
            break;
        case 20:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(3-r)*4+(2-s)] += result;
                }
            }
            break;
        case 21:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(3-r)*4+(3-s)] += result;
                }
            }
            break;
        case 22:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(3-r)*4+(4-s)] += result;
                }
            }
            break;
        case 23:
            for ( int r = 0; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(3-r)*4+(5-s)] += result;
                }
            }
            break;
        case 24:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(4-r)*4+(0-s)] += result;
                }
            }
            break;
        case 25:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(4-r)*4+(1-s)] += result;
                }
            }
            break;
        case 26:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(4-r)*4+(2-s)] += result;
                }
            }
            break;
        case 27:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(4-r)*4+(3-s)] += result;
                }
            }
            break;
        case 28:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(4-r)*4+(4-s)] += result;
                }
            }
            break;
        case 29:
            for ( int r = 1; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(4-r)*4+(5-s)] += result;
                }
            }
            break;
        case 30:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 1; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(5-r)*4+(0-s)] += result;
                }
            }
            break;
        case 31:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 2; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(5-r)*4+(1-s)] += result;
                }
            }
            break;
        case 32:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(5-r)*4+(2-s)] += result;
                }
            }
            break;
        case 33:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 0; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(5-r)*4+(3-s)] += result;
                }
            }
            break;
        case 34:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 1; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(5-r)*4+(4-s)] += result;
                }
            }
            break;
        case 35:
            for ( int r = 2; r < 3; r++) {
                for ( int s = 2; s < 3; s++) {
                    float result = v * temp_kernel[r*S+s];
                    temp_result[(5-r)*4+(5-s)] += result;
                }
            }
            break;

    }
}
__device__ void load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                           float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                           unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                           unsigned int tile_id,unsigned int tile_c_id){
    for(unsigned int c_id=warp_id;c_id<TC&&tile_c_id+c_id<SC;c_id+=WARPS_CONV){
        unsigned int end_index = c_lens[batch_id*SC*TILES_EACH_CHANNEL+(tile_c_id+c_id)*TILES_EACH_CHANNEL+tile_id];
        if(lane_id ==0){
            shared_lens[c_id] = end_index;
        }
        for(unsigned int id = lane_id;id<end_index;id+=32){
            shared_input[c_id*(TH+R-1)*(TW+S-1)+id] = values[batch_id*SC*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+
                                                             (tile_c_id+c_id)*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+tile_id*(TH+R-1)*(TW+S-1)+id];
            shared_ids[c_id*(TH+R-1)*(TW+S-1)+id] = ids[batch_id*SC*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+
                                                        (tile_c_id+c_id)*TILES_EACH_CHANNEL*(TH+R-1)*(TW+S-1)+tile_id*(TH+R-1)*(TW+S-1)+id];
        }
    }
}
__global__ void conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                       char * __restrict__ ids,
                       const float * __restrict__ kernel, float * __restrict__ outputs){
    __shared__ float input[TC*(TH+R-1)*(TW+S-1)];
    __shared__ char input_ids[TC*(TH+R-1)*(TW+S-1)];
    __shared__ unsigned int channel_lens[(TC)];

    const unsigned int batch_id = (blockIdx.x/(TCS*TILES_EACH_CHANNEL));
    const unsigned int t_id = (blockIdx.x - batch_id*TCS*TILES_EACH_CHANNEL)/TCS;
    const unsigned int tile_h_id = (t_id / TWS)*TH;
    const unsigned int tile_w_id = (t_id % TWS)*TW;
    const unsigned int index = blockIdx.x % (TCS);
    const unsigned int start_channel_index = index*TC;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    float data_array[9];
    float temp_result[TH*TW] = {0.0f};
    load_input_2_shared_memory(values,c_lens,ids,input,input_ids,channel_lens,warp_id,lane_id,batch_id,t_id,start_channel_index);
    __syncthreads();
    float v;
    unsigned int id;
    for(unsigned int n = threadIdx.x;n<SN;n+=BLK_DIM){
        for(unsigned int c=start_channel_index;c<start_channel_index+TC&&c<SC;c++){
            unsigned int abs_c = c - start_channel_index;
            unsigned int start_index = abs_c*(TH+R-1)*(TW+S-1);
            unsigned int end_index = start_index+channel_lens[abs_c];
            if(start_index == end_index){
                continue;
            }
            load_data_2_register(data_array,(c),kernel,n);
            unsigned int iters = end_index - start_index;
            for(unsigned int iter=0;iter<iters;iter++) {
                v = input[iter+start_index];
                id = input_ids[iter+start_index];
                switch_function(id,data_array,v,temp_result);
            }
        }
        for (unsigned int th = 0; th < TH; ++th) {
            for (unsigned int tw = 0; tw < TW; ++tw) {
                if (tile_h_id + th >= SHo || tile_w_id + tw >= SWo) {
                    continue;
                }
                atomicAdd(&outputs[batch_id * SN * SHo * SWo + (tile_h_id + th) * SWo * SN + (tile_w_id + tw) * SN +
                                   n],temp_result[(th * TW + tw)]);
            }
        }
        for(unsigned int i=0;i<TH*TW;++i){
            temp_result[i] = 0.0f;
        }
    }
}







