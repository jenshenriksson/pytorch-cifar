'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from pytorch_eval import pytorch_train, pytorch_test

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--type', '-t', default=0, type=int, help='0: No preprocess, 1: Augment, 2: Aug+Cosine')
parser.add_argument('--batch', '-b', default=128, type=int, help='Batch size')
parser.add_argument('--model', default='densenet', help='Which model to train?')
parser.add_argument('--name', '-n', default=None, type=str, help='Use a specific name for saving.')
parser.add_argument('--epochs', '-e', default=100, type=int, help='Number of epochs to run')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


if args.type == 0:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
elif args.type >= 1:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    raise RuntimeError('Wrong type {}'.format(args.type))


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

if args.model == 'vgg': net = VGG('VGG16')
if args.model == 'resnet': net = ResNet18()
if args.model == 'preact': net = PreActResNet18()
if args.model == 'googlenet': net = GoogLeNet()
if args.model == 'densenet': net = DenseNet121()
if args.model == 'resnetx29': net = ResNeXt29_2x64d()
if args.model == 'mobile': net = MobileNet()
if args.model == 'mobilev2': net = MobileNetV2()
if args.model == 'dpn': net = DPN92()
if args.model == 'shuffle': net = ShuffleNetG2()
if args.model == 'senet': net = SENet18()
if args.model == 'shufflev2': net = ShuffleNetV2(1)
if args.model == 'wrn28': net = Wide_ResNet(28, 10, 0.3, 10)
if args.model == 'wrn40': net = Wide_ResNet(40, 10, 0.3, 10)

if args.name is not None:
    args.model = args.name

net = net.to(device)
if device == 'cuda':
    if args.model == 'vgg19': net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}.t7'.format(args.model))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.type >= 2:
    steps = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if not os.path.isdir('results'):
    os.mkdir('results')

print('Using {} with settings of type: {}. '.format(args.model, args.type))

for epoch in range(start_epoch+1, start_epoch+1+args.epochs):

    if not os.path.isfile('results/{}-results.txt'.format(args.model)):
        with open('results/{}-results.txt'.format(args.model), 'a') as f:
            f.write("epoch accuracy testloss trainloss\n")

    loss = pytorch_train(epoch, net, trainloader, device, optimizer, criterion, testloader, args)
    acc, loss_test, best_acc = pytorch_test(epoch, net, trainloader, device, optimizer, criterion, testloader, args, best_acc)

    if args.type >= 2:
        scheduler.step()
        print(scheduler.get_lr())

        if epoch % steps == 0:
            print('Resetting scheduler.')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    with open('results/{}-results.txt'.format(args.model), 'a') as f:
        f.write("{} {:2.2f} {:.5f} {:.5f}\n".format(epoch, acc, loss_test, loss))




state = {
    'net': net.state_dict(),
    'acc': acc,
    'epoch': epoch,
}

torch.save(state, './checkpoint/{}.t7'.format(args.model))
