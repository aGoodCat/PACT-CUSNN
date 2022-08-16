

#ifndef RESNET152_CONV1_H
#define RESNET152_CONV1_H
#include "common.h"
class Conv1{
public:
    cudnnHandle_t convCudnn;
    cudnnTensorDescriptor_t convInputDescriptor;
};
#endif //RESNET152_CONV1_H
