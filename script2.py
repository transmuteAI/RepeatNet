import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

for model in ['rresnet_cifar_16_4']:
    for dataset in ['TINYIMNET']:
        for wa in ['linear', 'swish']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 200 --epochs 90 --save_weights --weight_activation {wa}')
            
for model in ['rresnet_cifar_16_8']:
    for dataset in ['TINYIMNET']:
        for wa in ['bireal', 'linear', 'swish']:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 200 --epochs 90 --save_weights --weight_activation {wa}')

for model in ['resnet_cifar_16_2', 'resnet_cifar_16_4', 'resnet_cifar_16_8']:
    for dataset in ['TINYIMNET']:
        os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 200 --epochs 90 --save_weights')
        
        
        
        
        