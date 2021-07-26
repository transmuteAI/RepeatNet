import os
os.system("export CUDA_VISIBLE_DEVICES=0")
for model in ['resnet_16_2', 'resnet_16_4', 'resnet_16_8']:
    for dataset in ['CIFAR100']:
        for wa in ['swish', 'bireal', 'linear']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 100 --weight_activation {wa} --epochs 120 --save_weights')