import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

for model in ['rresnet_cifar_16_8']:
    for dataset in ['CIFAR100']:
        for wa in ['linear', 'swish']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 100 --epochs 160 --save_weights --weight_activation {wa}')

for model in ['resnet_cifar_16_2', 'resnet_cifar_16_4', 'resnet_cifar_16_8']:
    for dataset in ['CIFAR100']:
        os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 100 --epochs 160 --save_weights')

for model in ['rresnet_cifar_16_2', 'rresnet_cifar_16_4', 'rresnet_cifar_16_8']:
    for dataset in ['CIFAR10']:
        for wa in ['bireal', 'linear', 'swish']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 10 --epochs 160 --save_weights --weight_activation {wa}')

for model in ['resnet_cifar_16_1', 'resnet_cifar_16_2', 'resnet_cifar_16_4', 'resnet_cifar_16_8']:
    for dataset in ['CIFAR10']:
        os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 10 --epochs 160 --save_weights')