import os
os.system("export CUDA_VISIBLE_DEVICES=1")
for model in ['resnet_16_2', 'resnet_16_4', 'resnet_16_8']:
    for dataset in ['CIFAR100']:
        for dr in [0.1, 0.3, 0.5]:
            os.system(f'python main.py --model_name {model} --dataset {dataset} --num_classes 100 --weight_activation static_drop --epochs 120 --save_weights --drop_rate {dr}')