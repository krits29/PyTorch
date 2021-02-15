# Image Classification
Existing knowledge of PyTorch and linear regression can be implemented for a new problem: *image classification*. The training dataset will be the MNIST Handwritten Digits Database. It consists of several grayscale images of handwritten digits (0 to 9) and has a label for each image indicating the digit it reperesents. 

Here are some of the images in the MNIST dataset:

![mnist](https://i.imgur.com/CAYnuo1.jpg)

First things first, install and import `torch` and `torchvision`. 

`torchvision` contains some utilities for working with image data. It also provides helped classes to download and import datasets like MNIST automatically. 
```
import torch
import torchvision
from torchvision.datasets import MNIST
```
Download training dataset
```
dataset = MNIST(root='data/', download=True)
```
Upon checking the size of the dataset `len(dataset)` it shown that there are 60,000 images. These 60,000 images will be used to train the model. 

There is also an additional 'test set' of 10,000 images which will be used for evaluating the model and reporting metrics. This set can be created using `MNIST` class and passing `train = False` to the constructor.
```
test_dataset = MNIST(root='data/', train=False)
```
Here's a sample element from the training dataset `dataset[0]`. It contains a 28 by 28 pixel image and a label. 
```
(<PIL.Image.Image image mode=L size=28x28 at 0x7F625B9FD710>, 5)
```
In order to view the image, the `matplotlib` library can be used (also used for plotting and graphing data science in Python)
```
import matplotlib.pyplot as plt
%matplotlib inline
```
Here is one of the images from the dataset:
```
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)
```
Label: 5

![mnistimg](https://miro.medium.com/max/704/1*GlZuwzO-dMXxyEHO65Ha_A.png)

These images are relatively small in size, and it is challengeing to recognize the digits even to the human eye. 

Since PyTorch doesn't know how to work with images, they must be converted into tensors. This can be done by specifying a transform while creating the dataset. The `torchvision.transforms`module contains many predefined transformation functions that can be applied to the images as they are loaded. The `ToTenser` will transform the images into PyTorch tensors. 
```
import torchvision.transforms as transforms
```
MNIST Dataset (images and labels)
```
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())
```
Convert into a 1x28x28 tensor:
```
img_tensor, label = dataset[0]
```
