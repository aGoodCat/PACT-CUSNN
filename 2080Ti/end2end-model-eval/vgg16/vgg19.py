import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle
import codecs
from tensorflow.keras.utils import plot_model
from random import shuffle
batch_size = 4096
save_layers = ['block1_conv1','block1_conv2','block1_pool','block2_conv1','block2_conv2','block2_pool',
               'block3_conv1','block3_conv2','block3_conv3','block3_conv4','block3_pool','block4_conv1',
               'block4_conv2','block4_conv3','block4_conv4','block4_pool','block5_conv1','block5_conv2','block5_conv3']
print(save_layers)
net = tf.keras.applications.VGG16(include_top=True, weights='imagenet',
                                  input_tensor=None, input_shape=None,
                                  pooling=None, classes=1000)
print(net.summary())
for layer in net.layers:
    print(layer.name, len(layer.get_weights()))
    if len(layer.get_weights()) == 2:
        conv_weight = layer.get_weights()[0]
        conv_bias = layer.get_weights()[1]
        conv_weight = np.asarray(conv_weight)
        conv_bias = np.asarray(conv_bias)
        #print(conv_bias.shape,conv_weight.dtype)
        conv_weight = conv_weight.flatten()
        #print(conv_weight.shape,conv_weight.dtype)
        weight = np.concatenate((conv_weight, conv_bias), axis=0)
        weight_file = './weights/{}.bin'.format(layer.name)
        weight.tofile(weight_file)


