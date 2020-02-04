#!/usr/bin/bash
# Training runs for SEAA paper: DenseNet and VGG16 trained in three different ways. 

###### VGG16 #######
# First: No augmentation 
python36 main.py --lr 0.1 --model vgg --epochs 300 --type 0 --batch 128 --name VGG_seaa_clean

# Second: Augmentation 
python36 main.py --lr 0.1 --model vgg --epochs 300 --type 1 --batch 128 --name VGG_seaa_augm

# Third: Augmentation + Variable learning rate  
python36 main.py --lr 0.1 --model vgg --epochs 100 --type 1 --batch 128 --name VGG_seaa_augm_lr
python36 main.py --lr 0.01 --model vgg --epochs 100 --type 1 --batch 128 --name VGG_seaa_augm_lr
python36 main.py --lr 0.001 --model vgg --epochs 100 --type 1 --batch 128 --name VGG_seaa_augm_lr


###### DenseNet_121 #######
# First: No augmentation 
python36 main.py --lr 0.1 --model densenet --epochs 300 --type 0 --batch 128 --name DenseNet_seaa_clean

# Second: Augmentation 
python36 main.py --lr 0.1 --model densenet --epochs 300 --type 1 --batch 128 --name DenseNet_seaa_augm

# Third: Augmentation + Variable learning rate  
python36 main.py --lr 0.1 --model vgg --epochs 100 --type 1 --batch 128 --name DenseNet_seaa_augm_lr
python36 main.py --lr 0.01 --model vgg --epochs 100 --type 1 --batch 128 --name DenseNet_seaa_augm_lr
python36 main.py --lr 0.001 --model vgg --epochs 100 --type 1 --batch 128 --name DenseNet_seaa_augm_lr

