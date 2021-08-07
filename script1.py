import os

for model in ['resnet_18_1', 'resnet_18_2', 'resnet_cifar_18_4']:
    for dataset in ['IMNET']:
        os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 100 --epochs 90 --save_weights --datapath "/home/darya1/other_projects/Data/Imagenet" --gpus 4')

for model in ['rresnet_cifar_18_2', 'rresnet_cifar_18_4']:
    for dataset in ['IMNET']:
        for wa in ['bireal']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 10 --epochs 90 --save_weights --weight_activation {wa} --datapath "/home/darya1/other_projects/Data/Imagenet" --gpus 4')
