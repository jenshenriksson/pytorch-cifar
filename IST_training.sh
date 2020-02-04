#!/usr/bin/bash
# Training runs for IST-Journal submission:
# Training consist of Four Models, trained with augmentation and variable learning rate. 

###### VGG16 #######
# Augmentation and variable learning rate
python36 main.py --lr 0.1 --model vgg --epochs 100 --type 1 --batch 128 --name VGG_augm_lr
python36 main.py --lr 0.01 --model vgg --epochs 50 --resume --type 1 --batch 128 --name VGG_augm_lr
python36 main.py --lr 0.001 --model vgg --epochs 50 --resume --type 1 --batch 128 --name VGG_augm_lr


###### densenet #######
# Augmentation and variable learning rate
python36 main.py --lr 0.1 --model densenet --epochs 100 --type 1 --batch 128 --name DenseNet_augm_lr
python36 main.py --lr 0.01 --model densenet --epochs 50 --resume --type 1 --batch 128 --name DenseNet_augm_lr
python36 main.py --lr 0.001 --model densenet --epochs 50 --resume --type 1 --batch 128 --name DenseNet_augm_lr


###### WRN-28-10 #######
# Augmentation and variable learning rate
python36 main.py --lr 0.1 --model wrn28 --epochs 100 --type 1 --batch 128 --name WRN28_augm_lr
python36 main.py --lr 0.01 --model wrn28 --epochs 50 --resume --type 1 --batch 128 --name WRN28_augm_lr
python36 main.py --lr 0.001 --model wrn28 --epochs 50 --resume --type 1 --batch 128 --name WRN28_augm_lr


###### WRN-40-10 #######
# Augmentation and variable learning rate
python36 main.py --lr 0.1 --model wrn40 --epochs 100 --type 1 --batch 128 --name WRN40_augm_lr
python36 main.py --lr 0.01 --model wrn40 --epochs 50 --resume --type 1 --batch 128 --name WRN40_augm_lr
python36 main.py --lr 0.001 --model wrn40 --epochs 50 --resume --type 1 --batch 128 --name WRN40_augm_lr

