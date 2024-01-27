"""
This is a Visual Prompt Tuning practice created by Heejun Park.
A ViT-B/16 pre-trained on ImageNet21k is going to be used.
It will be fine-tuned using a prompt tuning method.
The dataset required for fine-tuning will be CIFAR-10.
Let's begin!  
"""

"""
Step 1: Import necessary libraries
"""
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10

from VPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--root', type=str, default='/root/datasets/ViT_practice/cifar10') # This is where the dataset is downloaded
    parser.add_argument('--log_dir', type=str, default='./model') # define the path used for storing the saved models
    parser.add_argument('--name', type=str, default='vit_cifar10')
    parser.add_argument('--rank', type=int, default=0)
    ops = parser.parse_args()
    print(ops.epoch)

    """
    Step 2: Load CIFAR-10 dataset
    """
    transform_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)),
        ])
    train_set = CIFAR10(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR10(root=ops.root,
                        train=False,
                        download=True,
                        transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set,
                                shuffle=True,
                                batch_size=ops.batch_size)

    test_loader = DataLoader(dataset=test_set,
                                shuffle=False,
                                batch_size=ops.batch_size)
    
    

if __name__ == '__main__':
    main()