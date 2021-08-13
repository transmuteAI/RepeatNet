import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for model in ['resnet_cifar_16_1', 'resnet_cifar_16_4', 'resnet_cifar_16_8']:
    for dataset in ['TINYIMNET']:
        os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 200 --epochs 90')
