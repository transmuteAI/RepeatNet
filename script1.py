import os
os.system("export CUDA_VISIBLE_DEVICES=0")
for model in ['rresnet_cifar_16_1', 'rresnet_cifar_16_2', 'rresnet_cifar_16_4', 'rresnet_cifar_16_8']:
    for dataset in ['TINYIMNET']:
        for wa in ['bireal', 'linear']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 200 --epochs 120 --save_weights --weight_activation {wa}')