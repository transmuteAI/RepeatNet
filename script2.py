import os
os.system("export CUDA_VISIBLE_DEVICES=1")
for model in range(4, 12):
    for dataset in ['TINYIMNET']:
        os.system(f'python main.py --model_name VGG{model} --dataset {dataset} --num_classes 200 --epochs 120 --save_weights --weight_activation None')