import torch
import datasets
import os
import sys
import pathlib
import subprocess


if __name__=='__main__':
    # gpu_info = !nvidia-smi
    # gpu_info = '\n'.join(gpu_info)
    # if gpu_info.find('failed') >= 0:
    #     print('Not connected to a GPU')
    # else:
    #     print(gpu_info)
    #get gpu info for python script
    #check for packages in requirements.txt and versions
    # with open('requirements.txt') as f:
    #     requirements = f.read().splitlines()
        
    # for package in requirements:
    #     print(package)
    #     print(subprocess.run([sys.executable, '-m', 'pip', 'show', package], stdout=subprocess.PIPE).stdout.decode('utf-8'))
        
    # gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # print(gpu_info)
    # print(torch.cuda.is_available())

            