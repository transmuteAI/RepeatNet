import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for model in [4,5,7,8,9]:
    os.system(f'python main.py --model_name VGG{model} --dataset CIFAR100 --num_classes 100 --epochs 160')

for model in [4,8,9,10]:
    os.system(f'python main.py --model_name VGG{model} --dataset CIFAR10 --num_classes 10 --epochs 160')

