import subprocess as sp
from multiprocessing.dummy import Pool
import itertools
import sys, os
import argparse
import random

# adapt these to you setup
NR_GPUS = 4
NR_PROCESSES = 4

cnt = -1


def call_script(scripts):
    global cnt
    model, dataset, num_classes, wa, epochs = scripts
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    crt_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cnt += 1
    gpu = cnt % NR_GPUS
    crt_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(scripts)
    sp.call([sys.executable, 'main.py', '--model_name', str(model), '--dataset', str(dataset), '--num_classes', str(num_classes), '--epochs', str(epochs), '--weight_activation', wa], env=crt_env)


if __name__ == '__main__':
    pool = Pool(NR_PROCESSES)
    models = [f'rep_vgg_{i}' for i in range(4,11)]
    was = ['linear', 'bireal', 'swish']
    
    scripts = list(itertools.product(models, ['TINYIMNET'], [200], was, [90]))
    models = [f'VGG{i}' for i in range(4,12)]
    scripts += list(itertools.product(models, ['TINYIMNET'], [200], ['linear'], [90]))
    scripts += list(itertools.product(models, ['CIFAR100'], [100], ['linear'], [160]))
    scripts += list(itertools.product(models, ['CIFAR10'], [10], ['linear'], [160]))
    
    pool.map(call_script, scripts)
    pool.close()
    pool.join()
    
