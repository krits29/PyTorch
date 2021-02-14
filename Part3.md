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

There is also an additional "test-set" of 10,000 images which will be used for evaluating hte model and reporting metrics. This set can be created using `MNIST` class and passing `train = False` to the constructor.


