import codecs
import os,sys
import subprocess
import time
import shutil

root_dir = os.getcwd()

print("starting build")
energy_densenet169 = os.path.join(root_dir,'end2end-energy-eval/densenet169/build')
energy_densenet169_cusnn = os.path.join(root_dir,'end2end-energy-eval/densenet169Scnn/build')
energy_densenet201 = os.path.join(root_dir,'end2end-energy-eval/densenet201/build')
energy_densenet201_cusnn = os.path.join(root_dir,'end2end-energy-eval/densenet201Scnn/build')
energy_resnet50 = os.path.join(root_dir,'end2end-energy-eval/resnet50/build')
energy_resnet50_cusnn = os.path.join(root_dir,'end2end-energy-eval/resnet50Scnn/build')
energy_resnet152 = os.path.join(root_dir,'end2end-energy-eval/resnet152/build')
energy_resnet152_cusnn = os.path.join(root_dir,'end2end-energy-eval/resnet152Scnn/build')
energy_vgg16 = os.path.join(root_dir,'end2end-energy-eval/vgg16/build')
energy_vgg16_cusnn = os.path.join(root_dir,'end2end-energy-eval/vgg16Scnn/build')
energy_vgg19 = os.path.join(root_dir,'end2end-energy-eval/vgg19/build')
energy_vgg19_cusnn = os.path.join(root_dir,'end2end-energy-eval/vgg19Scnn/build')

model_densenet169 = os.path.join(root_dir,'end2end-model-eval/densenet169/build')
model_densenet169_cusnn = os.path.join(root_dir,'end2end-model-eval/densenet169Scnn/build')
model_densenet201 = os.path.join(root_dir,'end2end-model-eval/densenet201/build')
model_densenet201_cusnn = os.path.join(root_dir,'end2end-model-eval/densenet201Scnn/build')
model_resnet50 = os.path.join(root_dir,'end2end-model-eval/resnet50/build')
model_resnet50_cusnn = os.path.join(root_dir,'end2end-model-eval/resnet50Scnn/build')
model_resnet152 = os.path.join(root_dir,'end2end-model-eval/resnet152/build')
model_resnet152_cusnn = os.path.join(root_dir,'end2end-model-eval/resnet152Scnn/build')
model_vgg16 = os.path.join(root_dir,'end2end-model-eval/vgg16/build')
model_vgg16_cusnn = os.path.join(root_dir,'end2end-model-eval/vgg16Scnn/build')
model_vgg19 = os.path.join(root_dir,'end2end-model-eval/vgg19/build')
model_vgg19_cusnn = os.path.join(root_dir,'end2end-model-eval/vgg19Scnn/build')

cudnn_exec_pre_path = os.path.join(root_dir,'layers_eval/eval_cudnn/')
cudnn_exec_path = os.path.join(root_dir,'layers_eval/eval_cudnn/build')
cusnn_exec_path_densenet201 = os.path.join(root_dir,'layers_eval/densenet201Scnn/build')
cusnn_exec_path_resnet152 = os.path.join(root_dir,'layers_eval/resnet152Scnn/build')

if os.path.exists(cusnn_exec_path_densenet201):
    shutil.rmtree(cusnn_exec_path_densenet201)
os.mkdir(cusnn_exec_path_densenet201)
os.chdir(cusnn_exec_path_densenet201)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(cusnn_exec_path_resnet152):
    shutil.rmtree(cusnn_exec_path_resnet152)
os.mkdir(cusnn_exec_path_resnet152)
os.chdir(cusnn_exec_path_resnet152)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(cudnn_exec_path):
    shutil.rmtree(cudnn_exec_path)
os.mkdir(cudnn_exec_path)
os.chdir(cudnn_exec_path)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])



if os.path.exists(model_densenet169):
    shutil.rmtree(model_densenet169)
os.mkdir(model_densenet169)
os.chdir(model_densenet169)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])


if os.path.exists(model_densenet169_cusnn):
    shutil.rmtree(model_densenet169_cusnn)
os.mkdir(model_densenet169_cusnn)
os.chdir(model_densenet169_cusnn)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_densenet201):
    shutil.rmtree(model_densenet201)
os.mkdir(model_densenet201)
os.chdir(model_densenet201)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])


if os.path.exists(model_densenet201_cusnn):
    shutil.rmtree(model_densenet201_cusnn)
os.mkdir(model_densenet201_cusnn)
os.chdir(model_densenet201_cusnn)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_resnet50):
    shutil.rmtree(model_resnet50)
os.mkdir(model_resnet50)
os.chdir(model_resnet50)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_resnet50_cusnn):
    shutil.rmtree(model_resnet50_cusnn)
os.mkdir(model_resnet50_cusnn)
os.chdir(model_resnet50_cusnn)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_resnet152):
    shutil.rmtree(model_resnet152)
os.mkdir(model_resnet152)
os.chdir(model_resnet152)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])


if os.path.exists(model_resnet152_cusnn):
    shutil.rmtree(model_resnet152_cusnn)
os.mkdir(model_resnet152_cusnn)
os.chdir(model_resnet152_cusnn)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_vgg16):
    shutil.rmtree(model_vgg16)
os.mkdir(model_vgg16)
os.chdir(model_vgg16)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_vgg16_cusnn):
    shutil.rmtree(model_vgg16_cusnn)
os.mkdir(model_vgg16_cusnn)
os.chdir(model_vgg16_cusnn)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_vgg19):
    shutil.rmtree(model_vgg19)
os.mkdir(model_vgg19)
os.chdir(model_vgg19)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

if os.path.exists(model_vgg19_cusnn):
    shutil.rmtree(model_vgg19_cusnn)
os.mkdir(model_vgg19_cusnn)
os.chdir(model_vgg19_cusnn)
subprocess.run(["cmake",".."])
subprocess.run(["make","-j"])

print("end building")
time.sleep(2)



time.sleep(2)

print('starting compare end to end performance cuDNN VS cuSNN...')
print('starting evaluate densenet169 time consumption(cuDNN) , unit is milli-sec(ms)...')
os.chdir(model_densenet169)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate densenet169 time consumption(cuSNN) , unit is milli-sec(ms)...')
os.chdir(model_densenet169_cusnn)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate densenet201 time consumption(cuDNN) , unit is milli-sec(ms)...')
os.chdir(model_densenet201)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate densenet201 time consumption(cuSNN) , unit is milli-sec(ms)...')
os.chdir(model_densenet201_cusnn)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate resnet50 time consumption(cuDNN) , unit is milli-sec(ms)...')
os.chdir(model_resnet50)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate resnet50 time consumption(cuSNN) , unit is milli-sec(ms)...')
os.chdir(model_resnet50_cusnn)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate resnet152 time consumption(cuDNN) , unit is milli-sec(ms)...')
os.chdir(model_resnet152)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate resnet152 time consumption(cuSNN) , unit is milli-sec(ms)...')
os.chdir(model_resnet152_cusnn)
subprocess.run(["./test","../../../../sample_images/"])

#please uncomment below to test VGG
'''
print('starting evaluate vgg16 time consumption(cuDNN) , unit is milli-sec(ms)...')
os.chdir(model_vgg16)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate vgg16 time consumption(cuSNN) , unit is milli-sec(ms)...')
os.chdir(model_vgg16_cusnn)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate vgg19 time consumption(cuDNN) , unit is milli-sec(ms)....')
os.chdir(model_vgg19)
subprocess.run(["./test","../../../../sample_images/"])
print('starting evaluate vgg19 time consumption(cuSNN) , unit is milli-sec(ms)...')
os.chdir(model_vgg19_cusnn)
subprocess.run(["./test","../../../../sample_images/"])
'''
time.sleep(2)
print('starting evaluate layerwise speedup for densenet201 and resnet152...')
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

