import codecs
import os,sys
import subprocess


root_dir = os.getcwd()

cudnn_exec_pre_path = os.path.join(root_dir,'layers_eval/eval_cudnn/')
cusnn_exec_path_densenet201 = os.path.join(root_dir,'layers_eval/densenet201Scnn/build')
cusnn_exec_path_resnet152 = os.path.join(root_dir,'layers_eval/resnet152Scnn/build')
os.chdir(cudnn_exec_pre_path)
reader = codecs.open('conv_layer_id.csv','r','utf-8')
lines = reader.readlines()
if os.path.exists('cudnn_layers.txt'):
    os.remove('cudnn_layers.txt')
print('Step1: Run densenet201&resnet152 layers evaluation cuDNN')
for line in lines:
    parts = line.split(',')
    network = parts[0]
    id = int(parts[1])
    C = int(parts[2])
    H = int(parts[3])
    W = int(parts[4])
    N = int(parts[5])
    id = id + 1
    for b in [1]:
        subprocess.run(["./build/test","{}".format(b),"{}".format(C),"{}".format(H),"{}".format(W),"{}".format(N),"{}".format(network),
                        "{}".format(id)])

reader = codecs.open('cudnn_layers.txt','r','utf-8')
densenet_layers = 0
densenet_layers_cudnn = {}
densenet_layers_cusnn = {}
resnet_layers = 0
resnet_layers_cudnn = {}
resnet_layers_cusnn = {}

lines = reader.readlines()
for line in lines:
    parts = line.split(',')
    network = parts[0]
    layer_id = int(parts[1])
    run_time = float(parts[2])
    if network == 'densenet201':
        densenet_layers +=1
        densenet_layers_cudnn[layer_id] = run_time
    if network == 'resnet152':
        resnet_layers += 1
        resnet_layers_cudnn[layer_id] = run_time
os.remove('cudnn_layers.txt')


os.chdir(cusnn_exec_path_densenet201)
print('Step2: Run densenet201&resnet152 layers evaluation cuSNN')
subprocess.run(["./test","../../../../sample_images/"])
reader = codecs.open('densenet201_layers.txt','r','utf-8')
lines = reader.readlines()
for line in lines:
    parts = line.split(',')
    layer_id = int(parts[0])
    run_time = float(parts[1])
    if run_time == 0.0:
        continue
    densenet_layers_cusnn[layer_id] = run_time
os.remove('densenet201_layers.txt')


os.chdir(cusnn_exec_path_resnet152)
subprocess.run(["./test","../../../../sample_images/"])
reader = codecs.open('resnet152_layers.txt','r','utf-8')
lines = reader.readlines()
for line in lines:
    parts = line.split(',')
    layer_id = int(parts[0])
    run_time = float(parts[1])
    if run_time == 0.0:
        continue
    resnet_layers_cusnn[layer_id] = run_time
os.remove('resnet152_layers.txt')

speedup_densenet = 0.0
for i in range(densenet_layers):
    data_key = i + 1
    cusnn_time = densenet_layers_cusnn[data_key]
    cudnn_time = densenet_layers_cudnn[data_key]
    speedup_densenet += cudnn_time/cusnn_time
    print('densenet201 layer {}, speedup {}'.format(data_key, cudnn_time/cusnn_time))
print('Average speedup across all layers in densenet201 is {}'.format(speedup_densenet/densenet_layers))

speedup_resnet = 0.0
for i in range(resnet_layers):
    data_key = i + 1
    cusnn_time = resnet_layers_cusnn[data_key]
    cudnn_time = resnet_layers_cudnn[data_key]
    speedup_resnet += cudnn_time/cusnn_time
    print('resnet152 layer {}, speedup {}'.format(data_key, cudnn_time/cusnn_time))
print('Average speedup across all layers in resnet152 is {}'.format(speedup_resnet/resnet_layers))
