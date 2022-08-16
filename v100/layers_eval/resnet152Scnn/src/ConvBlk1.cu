#include "../inc/convBlk.h"
#define MEASUE_CUSNN true
ConvBlk1::ConvBlk1(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight,
                   string bn2Weight,string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2){
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,256,0,1,1,1,conv0Weight);
    conv1.initialize(B,C,H,W,64,0,1,1,1,conv1Weight);
    conv2.initialize(B,C,H,W,64,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,C,H,W,64,1,3,3,1,conv2Weight);
    conv3.initialize(B,C,H,W,256,0,1,1,1,conv3Weight);
    bn0.initialize(B,256,H,W,bn0Weight);
    bn1.initialize(B,64,H,W,bn1Weight);
    bn2.initialize(B,64,H,W,bn2Weight);
    bn3.initialize(B,256,H,W,bn3Weight);
    add.initialize(B,256,H,W);
    activation64.initialize(B,64,H,W);
    activation64_cudnn.initialize(B,64,H,W);
    activation.initialize(B,256,H,W);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk1::forward(float *input){
    float *block1_1_conv = conv1.forward(input);
    float *block1_1_bn = bn1.forward(block1_1_conv);
    float *block1_1_relu;
    float *block1_2_conv;
    if(sparse){
        block1_1_relu = activation64.forward(block1_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block1_2_conv = conv2.forward(block1_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block1_2_conv = conv2_cudnn.forward(block1_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
    }else{
        block1_1_relu = activation64_cudnn.forward(block1_1_bn);
        block1_2_conv = conv2_cudnn.forward(block1_1_relu);
    }

    float *block1_2_bn = bn2.forward(block1_2_conv);
    float *block1_2_relu = activation64_cudnn.forward(block1_2_bn);

    float *block1_3_conv = conv3.forward(block1_2_relu);
    float *block1_3_bn = bn3.forward(block1_3_conv);

    float *block1_0_conv = conv0.forward(input);
    float *block1_0_bn = bn0.forward(block1_0_conv);

    float *block1_add = add.forward(block1_3_bn,block1_0_bn);
    float *relu = activation.forward(block1_add);
    return relu;
}
ConvBlk2::ConvBlk2(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string conv1Weight, string bn1Weight,
                   string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv1.initialize(B,C,H,W,64,0,1,1,1,conv1Weight);
    conv2.initialize(B,conv1.N,conv1.hOut,conv1.wOut,64,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,conv1.N,conv1.hOut,conv1.wOut,64,1,3,3,1,conv2Weight);
    conv3.initialize(B,conv2.N,conv2.hOut,conv2.wOut,256,0,1,1,1,conv3Weight);
    bn1.initialize(B,64,conv1.hOut,conv1.wOut,bn1Weight);
    bn2.initialize(B,64,conv2.hOut,conv2.wOut,bn2Weight);
    bn3.initialize(B,256,conv3.hOut,conv3.wOut,bn3Weight);
    add.initialize(B,256,H,W);
    activation64.initialize(B,64,conv1.hOut,conv1.wOut);
    activation64_cudnn.initialize(B,64,conv1.hOut,conv1.wOut);
    activation.initialize(B,256,conv3.hOut,conv3.wOut);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk2::forward(float *input){
    float *block2_1_conv = conv1.forward(input);
    float *block2_1_bn = bn1.forward(block2_1_conv);
    float *block2_1_relu;
    float *block2_2_conv;
    if(sparse){
        block2_1_relu = activation64.forward(block2_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block2_2_conv = conv2.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block2_2_conv = conv2_cudnn.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }

    }else{
        block2_1_relu = activation64_cudnn.forward(block2_1_bn);
        block2_2_conv = conv2_cudnn.forward(block2_1_relu);
    }
    float *block2_2_bn = bn2.forward(block2_2_conv);
    float *block2_2_relu = activation64_cudnn.forward(block2_2_bn);

    float *block2_3_conv = conv3.forward(block2_2_relu);
    float *block2_3_bn = bn3.forward(block2_3_conv);

    float *block2_add = add.forward(block2_3_bn,input);
    float *block2_out = activation.forward(block2_add);
    return block2_out;
}
ConvBlk3::ConvBlk3(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight,
                   string bn2Weight,string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,512,0,1,1,2,conv0Weight);
    conv1.initialize(B,C,H,W,128,0,1,1,2,conv1Weight);
    conv2.initialize(B,128,H/2,W/2,128,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,128,H/2,W/2,128,1,3,3,1,conv2Weight);
    conv3.initialize(B,128,H/2,W/2,512,0,1,1,1,conv3Weight);
    bn0.initialize(B,512,H/2,W/2,bn0Weight);
    bn1.initialize(B,128,H/2,W/2,bn1Weight);
    bn2.initialize(B,128,H/2,W/2,bn2Weight);
    bn3.initialize(B,512,H/2,W/2,bn3Weight);
    add.initialize(B,512,H/2,W/2);
    activation64.initialize(B,128,H/2,W/2);
    activation64_cudnn.initialize(B,128,H/2,W/2);
    activation.initialize(B,512,H/2,W/2);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk3::forward(float *input){
    float *block3_1_conv = conv1.forward(input);
    float *block3_1_bn = bn1.forward(block3_1_conv);
    float *block3_1_relu;
    float *block3_2_conv;
    if(sparse){
        block3_1_relu = activation64.forward(block3_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);
        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block3_2_conv = conv2.forward(block3_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block3_2_conv = conv2_cudnn.forward(block3_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
    }else{
        block3_1_relu = activation64_cudnn.forward(block3_1_bn);
        block3_2_conv = conv2_cudnn.forward(block3_1_relu);
    }
    float *block3_2_bn = bn2.forward(block3_2_conv);
    float *block3_2_relu = activation64_cudnn.forward(block3_2_bn);

    float *block3_3_conv = conv3.forward(block3_2_relu);
    float *block3_3_bn = bn3.forward(block3_3_conv);

    float *block3_0_conv = conv0.forward(input);
    float *block3_0_bn = bn0.forward(block3_0_conv);

    float *block3_add = add.forward(block3_3_bn,block3_0_bn);
    float *block3_out = activation.forward(block3_add);
    return block3_out;
}
ConvBlk4::ConvBlk4(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string conv1Weight, string bn1Weight,
                   string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv1.initialize(B,C,H,W,128,0,1,1,1,conv1Weight);
    conv2.initialize(B,128,H,W,128,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,128,H,W,128,1,3,3,1,conv2Weight);
    conv3.initialize(B,128,H,W,512,0,1,1,1,conv3Weight);
    bn1.initialize(B,128,H,W,bn1Weight);
    bn2.initialize(B,128,H,W,bn2Weight);
    bn3.initialize(B,512,H,W,bn3Weight);
    add.initialize(B,512,H,W);
    activation64.initialize(B,128,H,W);
    activation64_cudnn.initialize(B,128,H,W);
    activation.initialize(B,512,H,W);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk4::forward(float *input){
    float *block2_1_conv = conv1.forward(input);
    float *block2_1_bn = bn1.forward(block2_1_conv);
    float *block2_1_relu;
    float *block2_2_conv;
    if(sparse){
        block2_1_relu = activation64.forward(block2_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block2_2_conv = conv2.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block2_2_conv = conv2_cudnn.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }

    }else{
        block2_1_relu = activation64_cudnn.forward(block2_1_bn);
        block2_2_conv = conv2_cudnn.forward(block2_1_relu);
    }
    float *block2_2_bn = bn2.forward(block2_2_conv);
    float *block2_2_relu = activation64_cudnn.forward(block2_2_bn);

    float *block2_3_conv = conv3.forward(block2_2_relu);
    float *block2_3_bn = bn3.forward(block2_3_conv);

    float *block2_add = add.forward(block2_3_bn,input);
    float *block2_out = activation.forward(block2_add);
    return block2_out;
}
ConvBlk5::ConvBlk5(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight, string bn2Weight,
                   string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,1024,0,1,1,2,conv0Weight);
    conv1.initialize(B,C,H,W,256,0,1,1,2,conv1Weight);
    conv2.initialize(B,256,H/2,W/2,256,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,256,H/2,W/2,256,1,3,3,1,conv2Weight);
    conv3.initialize(B,256,H/2,W/2,1024,0,1,1,1,conv3Weight);
    bn0.initialize(B,1024,H/2,W/2,bn0Weight);
    bn1.initialize(B,256,H/2,W/2,bn1Weight);
    bn2.initialize(B,256,H/2,W/2,bn2Weight);
    bn3.initialize(B,1024,H/2,W/2,bn3Weight);
    add.initialize(B,1024,H/2,W/2);
    activation64.initialize(B,256,H/2,W/2);
    activation64_cudnn.initialize(B,256,H/2,W/2);
    activation.initialize(B,1024,H/2,W/2);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk5::forward(float *input){
    float *block3_1_conv = conv1.forward(input);
    float *block3_1_bn = bn1.forward(block3_1_conv);
    float *block3_1_relu;
    float *block3_2_conv;
    if(sparse){
        block3_1_relu = activation64.forward(block3_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block3_2_conv = conv2.forward(block3_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block3_2_conv = conv2_cudnn.forward(block3_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }

    }else{
        block3_1_relu = activation64_cudnn.forward(block3_1_bn);
        block3_2_conv = conv2_cudnn.forward(block3_1_relu);
    }
    float *block3_2_bn = bn2.forward(block3_2_conv);
    float *block3_2_relu = activation64_cudnn.forward(block3_2_bn);

    float *block3_3_conv = conv3.forward(block3_2_relu);
    float *block3_3_bn = bn3.forward(block3_3_conv);

    float *block3_0_conv = conv0.forward(input);
    float *block3_0_bn = bn0.forward(block3_0_conv);

    float *block3_add = add.forward(block3_3_bn,block3_0_bn);
    float *block3_out = activation.forward(block3_add);
    return block3_out;
}
ConvBlk6::ConvBlk6(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string conv1Weight, string bn1Weight,
                   string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv1.initialize(B,C,H,W,256,0,1,1,1,conv1Weight);
    conv2.initialize(B,256,H,W,256,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,256,H,W,256,1,3,3,1,conv2Weight);
    conv3.initialize(B,256,H,W,1024,0,1,1,1,conv3Weight);
    bn1.initialize(B,256,H,W,bn1Weight);
    bn2.initialize(B,256,H,W,bn2Weight);
    bn3.initialize(B,1024,H,W,bn3Weight);
    add.initialize(B,1024,H,W);
    activation64.initialize(B,256,H,W);
    activation64_cudnn.initialize(B,256,H,W);
    activation.initialize(B,1024,H,W);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk6::forward(float *input){
    float *block2_1_conv = conv1.forward(input);
    float *block2_1_bn = bn1.forward(block2_1_conv);
    float *block2_1_relu;
    float *block2_2_conv;
    if(sparse){
        block2_1_relu = activation64.forward(block2_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block2_2_conv = conv2.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block2_2_conv = conv2_cudnn.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
        /**/

    }else{
        block2_1_relu = activation64_cudnn.forward(block2_1_bn);
        block2_2_conv = conv2_cudnn.forward(block2_1_relu);
    }
    float *block2_2_bn = bn2.forward(block2_2_conv);
    float *block2_2_relu = activation64_cudnn.forward(block2_2_bn);

    float *block2_3_conv = conv3.forward(block2_2_relu);
    float *block2_3_bn = bn3.forward(block2_3_conv);

    float *block2_add = add.forward(block2_3_bn,input);
    float *block2_out = activation.forward(block2_add);
    return block2_out;
}
ConvBlk7::ConvBlk7(unsigned int b, unsigned int c, unsigned int h, unsigned int w, string conv0Weight, string bn0Weight,
                   string conv1Weight, string bn1Weight, string conv2Weight,
                   string bn2Weight,string conv3Weight,string bn3Weight, bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv0.initialize(B,C,H,W,2048,0,1,1,2,conv0Weight);
    conv1.initialize(B,C,H,W,512,0,1,1,2,conv1Weight);
    conv2.initialize(B,512,H/2,W/2,512,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,512,H/2,W/2,512,1,3,3,1,conv2Weight);
    conv3.initialize(B,512,H/2,W/2,2048,0,1,1,1,conv3Weight);
    bn0.initialize(B,2048,H/2,W/2,bn0Weight);
    bn1.initialize(B,512,H/2,W/2,bn1Weight);
    bn2.initialize(B,512,H/2,W/2,bn2Weight);
    bn3.initialize(B,2048,H/2,W/2,bn3Weight);
    add.initialize(B,2048,H/2,W/2);
    activation64.initialize(B,512,H/2,W/2);
    activation64_cudnn.initialize(B,512,H/2,W/2);
    activation.initialize(B,2048,H/2,W/2);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk7::forward(float *input){
    float *block3_1_conv = conv1.forward(input);
    float *block3_1_bn = bn1.forward(block3_1_conv);
    float *block3_1_relu;
    float *block3_2_conv;
    if(sparse){
        block3_1_relu = activation64.forward(block3_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block3_2_conv = conv2.forward(block3_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block3_2_conv = conv2_cudnn.forward(block3_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }

    }else{
        block3_1_relu = activation64_cudnn.forward(block3_1_bn);
        block3_2_conv = conv2_cudnn.forward(block3_1_relu);
    }
    float *block3_2_bn = bn2.forward(block3_2_conv);
    float *block3_2_relu = activation64_cudnn.forward(block3_2_bn);

    float *block3_3_conv = conv3.forward(block3_2_relu);
    float *block3_3_bn = bn3.forward(block3_3_conv);

    float *block3_0_conv = conv0.forward(input);
    float *block3_0_bn = bn0.forward(block3_0_conv);

    float *block3_add = add.forward(block3_3_bn,block3_0_bn);
    float *block3_out = activation.forward(block3_add);
    return block3_out;
}
ConvBlk8::ConvBlk8(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string conv1Weight, string bn1Weight,
                   string conv2Weight, string bn2Weight,string conv3Weight,string bn3Weight,bool sparse,int index, float *t1, float *t2) {
    B = b;
    H = h;
    C = c;
    W = w;
    conv1.initialize(B,C,H,W,512,0,1,1,1,conv1Weight);
    conv2.initialize(B,512,H,W,512,1,3,3,1,conv2Weight);
    conv2_cudnn.initialize(B,512,H,W,512,1,3,3,1,conv2Weight);
    conv3.initialize(B,512,H,W,2048,0,1,1,1,conv3Weight);
    bn1.initialize(B,512,H,W,bn1Weight);
    bn2.initialize(B,512,H,W,bn2Weight);
    bn3.initialize(B,2048,H,W,bn3Weight);
    add.initialize(B,2048,H,W);
    activation64.initialize(B,512,H,W);
    activation64_cudnn.initialize(B,512,H,W);
    activation.initialize(B,2048,H,W);
    this->sparse = sparse;
    this->index = index;
    this->t1 = t1;
    this->t2 = t2;

}
float * ConvBlk8::forward(float *input){
    float *block2_1_conv = conv1.forward(input);
    float *block2_1_bn = bn1.forward(block2_1_conv);
    float *block2_1_relu;
    float *block2_2_conv;
    if(sparse){
        block2_1_relu = activation64.forward(block2_1_bn);
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);

        float temp_time;
        if(MEASUE_CUSNN){
            cudaEventRecord(event_start);
            block2_2_conv = conv2.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t1[index - 1] +=temp_time;
        }else{
            cudaEventRecord(event_start);
            block2_2_conv = conv2_cudnn.forward(block2_1_relu);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&temp_time, event_start, event_stop);
            t2[index - 1] +=temp_time;
        }
    }else{
        block2_1_relu = activation64_cudnn.forward(block2_1_bn);
        block2_2_conv = conv2_cudnn.forward(block2_1_relu);
    }
    float *block2_2_bn = bn2.forward(block2_2_conv);
    float *block2_2_relu = activation64_cudnn.forward(block2_2_bn);

    float *block2_3_conv = conv3.forward(block2_2_relu);
    float *block2_3_bn = bn3.forward(block2_3_conv);

    float *block2_add = add.forward(block2_3_bn,input);
    float *block2_out = activation.forward(block2_add);
    return block2_out;
}
