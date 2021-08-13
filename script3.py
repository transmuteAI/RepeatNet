import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
for model in [4,5,7,8,10,11]:
    os.system(f'python main.py --model_name VGG{model} --dataset TINYIMNET --num_classes 200 --epochs 90')

for model in [4,6,7,8,9]:
    os.system(f'python main.py --model_name rep_vgg_{model} --dataset TINYIMNET --num_classes 200 --epochs 90 --weight_activation "bireal"')

