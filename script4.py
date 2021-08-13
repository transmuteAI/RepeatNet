import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
for model in [10]:
    os.system(f'python main.py --model_name rep_vgg_{model} --dataset TINYIMNET --num_classes 200 --epochs 90 --weight_activation "linear"')

for model in [5,6,8,10]:
    os.system(f'python main.py --model_name rep_vgg_{model} --dataset TINYIMNET --num_classes 200 --epochs 90 --weight_activation "swish"')
