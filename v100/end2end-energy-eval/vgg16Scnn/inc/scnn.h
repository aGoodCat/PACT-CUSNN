

#ifndef VGG16SCNN_SCNN_H
#define VGG16SCNN_SCNN_H
#include "common.h"
__device__ void conv_1_512_14_14_512_switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result);
__global__ void conv_1_512_14_14_512_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens);
__device__ void conv_1_512_14_14_512_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id);
__device__ void conv_1_512_14_14_512_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                                float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                                unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                                unsigned int tile_id,unsigned int tile_c_id);
__global__ void conv_1_512_14_14_512_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                            char * __restrict__ ids,
                                            const float * __restrict__ kernel, float * __restrict__ outputs);
__device__ void conv_1_512_28_28_512_switch_function( unsigned int switch_condition,float *temp_kernel,
                                                      float v,float *temp_result);
__global__ void conv_1_512_28_28_512_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens);
__device__ void conv_1_512_28_28_512_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id);
__device__ void conv_1_512_28_28_512_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                                float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                                unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                                unsigned int tile_id,unsigned int tile_c_id);
__global__ void conv_1_512_28_28_512_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                            char * __restrict__ ids,
                                            const float * __restrict__ kernel, float * __restrict__ outputs);
__global__ void conv_1_256_28_28_512_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                            char * __restrict__ ids,
                                            const float * __restrict__ kernel, float * __restrict__ outputs);
__device__ void conv_1_256_28_28_512_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                                float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                                unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                                unsigned int tile_id,unsigned int tile_c_id);
__device__ void conv_1_256_28_28_512_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id);
__global__ void conv_1_256_28_28_512_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens);
__device__ void conv_1_256_28_28_512_switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result);
__global__ void relu(float * __restrict__ values, unsigned int B,unsigned int C,unsigned int H,
                     unsigned int W,unsigned int *counter);

__device__ void conv_1_128_112_112_128_switch_function( unsigned int switch_condition,float *temp_kernel,
                                                      float v,float *temp_result);
__global__ void conv_1_128_112_112_128_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens);
__device__ void conv_1_128_112_112_128_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id);
__device__ void conv_1_128_112_112_128_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                                float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                                unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                                unsigned int tile_id,unsigned int tile_c_id);
__global__ void conv_1_128_112_112_128_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                            char * __restrict__ ids,
                                            const float * __restrict__ kernel, float * __restrict__ outputs);

__device__ void conv_1_64_224_224_64_switch_function( unsigned int switch_condition,float *temp_kernel,
                                                        float v,float *temp_result);
__global__ void conv_1_64_224_224_64_transform(float *matrix, float *d_nnz, char *ids, unsigned int *c_lens);
__device__ void conv_1_64_224_224_64_load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id);
__device__ void conv_1_64_224_224_64_load_input_2_shared_memory(float *values,unsigned int *c_lens,char *ids,
                                                                  float *shared_input,char *shared_ids,unsigned int *shared_lens,
                                                                  unsigned int warp_id,unsigned int lane_id,unsigned int batch_id,
                                                                  unsigned int tile_id,unsigned int tile_c_id);
__global__ void conv_1_64_224_224_64_conv2d(float * __restrict__ values, unsigned int * __restrict__ c_lens,
                                              char * __restrict__ ids,
                                              const float * __restrict__ kernel, float * __restrict__ outputs);
#endif //VGG16SCNN_SCNN_H
