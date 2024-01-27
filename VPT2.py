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