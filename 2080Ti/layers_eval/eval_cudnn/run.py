import codecs
import os,sys
import subprocess
reader = codecs.open('conv_layer_id.csv','r','utf-8')
lines = reader.readlines()
for line in lines:
    parts = line.split(',')
    network = parts[0]
    id = int(parts[1])
    C = int(parts[2])
    H = int(parts[3])
    W = int(parts[4])
    N = int(parts[5])
    id = id + 1
    if os.path.exists('densenet201_layers.txt'):
        os.remove('densenet201_layers.txt')
    if os.path.exists('cudnn_layers.txt'):
        os.remove('cudnn_layers.txt')
    if os.path.exists('resnet152_layers.txt'):
        os.remove('resnet152_layers.txt')
    print('Step1: Run densenet201&resnet152 layers evaluation cuDNN')
    for b in [1]:
        subprocess.run(["./build/test","{}".format(b),"{}".format(C),"{}".format(H),"{}".format(W),"{}".format(N),"{}".format(network),"{}".format(id)])
    print('Step2: Run densenet201&resnet152 layers evaluation cuSNN')
    subprocess.run(["../densenet201Scnn/build/test","../../../sample_images/"])
    subprocess.run(["../resnet152Scnn/build/test","../../../sample_images/"])
