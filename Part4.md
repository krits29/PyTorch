# Training Deep Neural Networks

It's quite challenging to improve the accuracy of a logistic regression model beyond 87%. Instead, another approach called a *feed-forward neural network* can be used which can capture non-linear relationships between inputs and targets.

## Preparing the Data
The MNIST dataset provides 28px by 28px of grayscale images of handwritten digits (0-9) along with labels for each iamge indicating which digit it represents.

![MNIST](https://i.imgur.com/CAYnuo1.jpg)

Import all the required modules and classes:
```
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline
```

Download the data and create a PyTorch dataset using the `MNIST` class from `torchvision.datasets`
```
dataset = MNIST(root='data/', download=True, transform=ToTensor())
```
The images are converted into PyTorch tensors with the shape `1x28x28` (color channels, width, height). Use `plt.imshow` to display the images. (need to `permute` the method to reorder the dimensions of the image first)
```
image, label = dataset[0]
print('image.shape:', image.shape)
plt.imshow(image.permute(1, 2, 0), cmap='gray')
print('Label:', label)
```
Label: 5
