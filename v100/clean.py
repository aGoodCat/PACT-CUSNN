import codecs
import os,sys
import subprocess
import time
import shutil

root_dir = os.getcwd()

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

if os.path.exists(cusnn_exec_path_resnet152):
    shutil.rmtree(cusnn_exec_path_resnet152)


if os.path.exists(cudnn_exec_path):
    shutil.rmtree(cudnn_exec_path)


if os.path.exists(energy_densenet169):
    shutil.rmtree(energy_densenet169)


if os.path.exists(energy_densenet169_cusnn):
    shutil.rmtree(energy_densenet169_cusnn)

if os.path.exists(energy_densenet201):
    shutil.rmtree(energy_densenet201)



if os.path.exists(energy_densenet201_cusnn):
    shutil.rmtree(energy_densenet201_cusnn)


if os.path.exists(energy_resnet50):
    shutil.rmtree(energy_resnet50)


if os.path.exists(energy_resnet50_cusnn):
    shutil.rmtree(energy_resnet50_cusnn)


if os.path.exists(energy_resnet152):
    shutil.rmtree(energy_resnet152)



if os.path.exists(energy_resnet152_cusnn):
    shutil.rmtree(energy_resnet152_cusnn)


if os.path.exists(energy_vgg16):
    shutil.rmtree(energy_vgg16)


if os.path.exists(energy_vgg16_cusnn):
    shutil.rmtree(energy_vgg16_cusnn)


if os.path.exists(energy_vgg19):
    shutil.rmtree(energy_vgg19)


if os.path.exists(energy_vgg19_cusnn):
    shutil.rmtree(energy_vgg19_cusnn)


if os.path.exists(model_densenet169):
    shutil.rmtree(model_densenet169)


if os.path.exists(model_densenet169_cusnn):
    shutil.rmtree(model_densenet169_cusnn)


if os.path.exists(model_densenet201):
    shutil.rmtree(model_densenet201)



if os.path.exists(model_densenet201_cusnn):
    shutil.rmtree(model_densenet201_cusnn)


if os.path.exists(model_resnet50):
    shutil.rmtree(model_resnet50)


if os.path.exists(model_resnet50_cusnn):
    shutil.rmtree(model_resnet50_cusnn)


if os.path.exists(model_resnet152):
    shutil.rmtree(model_resnet152)



if os.path.exists(model_resnet152_cusnn):
    shutil.rmtree(model_resnet152_cusnn)


if os.path.exists(model_vgg16):
    shutil.rmtree(model_vgg16)


if os.path.exists(model_vgg16_cusnn):
    shutil.rmtree(model_vgg16_cusnn)


if os.path.exists(model_vgg19):
    shutil.rmtree(model_vgg19)


if os.path.exists(model_vgg19_cusnn):
    shutil.rmtree(model_vgg19_cusnn)

