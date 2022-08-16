# pact-cuSNN-AE
# cuSNN Prepare
## Package requirements:
* cmake(>=3.10)
* CUDA-10.2(V100)
* CUDA-11.0(2080Ti)
* Python3
* Tensorflow(optional)
## Setup
    export cuda_path=''
    export cuda_lib_path=''
    cuda_path is the absolute path(directory) where has the cuda headers. For me it is '/usr/local/cuda/include'
    cuda_lib_path is the absolute path(directory) where has the cuda library files. For me it is '/usr/local/cuda/lib64'

# Build code & Run experiments on V100 machine
  * cd v100
  * python3 run_v100.py(build and run)
  ## experiment 1
  it will report total energy consumption characteristics of different networks(densenet169,densenet201,resnet50,resnet152) on the V100
machine for both cuDNN and cuSNN. The unit is millijouel.
  ## experiment 2
  In experiment 2, we present the full network evaluation of different networks for both cuDNN and cuSNN. For the entire network evaluation, we include the time for all non-convolution and convolution layers. We use cuSNN kernels for sparse layers. The rest of the layers (dense convolution, batchnorm, relu, etc.) are built using cuDNN. 
 ## experiment 3
  In experiment 3, we we demonstrate the effectiveness of our approach by comparing the run-time of individual layers(densenet201&resnet152).
  
## Entire network evaluation for densenet(121, 169, 201), resnet(101, 152)
For the entire network evaluation, we include the time for all non-convolution and convolution layers. We use cuSNN kernels for sparse layers. The rest of the layers (dense convolution, batchnorm, relu, etc.) are built using cuDNN. Our entire network inference code takes a binary file representing the input images as input and outputs inference time and energy cost (mj )for the entire network.
    
    #cuDNN
    cd 2080ti/densenet121/build 
    ./test ../../../data/a.bin 

    #cuSNN
    cd 2080ti/densenet121Scnn/build
    ./test ../../../data/a.bin

a.bin is the image file in binary format. The provided `convert_img_2_bin.py` can produce binary files from normal png and jpg format. 

## cuSNN per layer network evaluation for densenet(121, 169, 201), resnet(101, 152)
For the per-layer network evaluation, we extracted the input feature map corresponding to each layer. We tested it using 128 images on five networks. We provide two scripts to evaluate the layers for different batch sizes (1, 2, 4, 8). The scripts output data in the following format f `B, C, H, W, N, t1, t2, t3, t4` (B = batch size, C = input channel, H = height, W = width, N = output channel, t1-t3 corresponds to the execution time of different cuDNN GEMM algorithms, and t4 corresponds to the cuSNN execution time. 

The following scripts report per layer cuSNN execution time on 2080tiand V100 machines.

    python run_2080ti_layers.py
    python run_v100_layers.py
## How to generate generate cuSNN convolution kernel for other machines/other convolution shapes (batch, height, width, input channel, output channel)
    cd sparse_conv_kernel_generator
    python code_generator.py
    
![Alt text](./sample.png?raw=true "Title")    
    
# Benchmarking platform and Dataset 

## Machine 1: 
* GPU: Nvidia GTX 2080 Ti (68 SMs, 11 GB)
* OS:  Ubuntu 20.04 LTS
* CUDA: 11.0
* cuDNN: 9.0.2

## Machine 2: 
* GPU: Nvidia Volta V100(84 SMs, 32GB)
* OS:   Ubuntu 18.04.4 LTS
* CUDA: 10.2
* cuDNN: 7.6.5

## Dataset:
* Imagenet - ILSVRC2012

# External Links
* Liang et al.: [https://github.com/terrance-liang/darknet-modified](url)
* Submanifold Sparse Convolutional Networks: [https://github.com/facebookresearch/SparseConvNet](url)
