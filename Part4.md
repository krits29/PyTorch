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
![MNISTexample](https://user-images.githubusercontent.com/52376448/63792062-bba44500-c937-11e9-9747-e048df95e1a6.png)

Label: 5

**Validation/Training Sets**

By using the `random_split` helper funciton, 10000 images can be set aside for the validation set.
```
val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
```
So now the training set contains 50000 images and the validation set contains 10000 images.

**Dataloaders**

First set the batch size, and then create the PyTorch data loaders for the training and validation sets.
```
batch_size=128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
```
## Layers

To improve upon logistic regression, a neural network with two layers (*a hidden layer* and an *output layer*) with an *activation function*  between the two layers can be created.

How it works:
* Instead of using a single `nn.Linear` object to transform a batch of inputs into a batch of outputs (pixel intensities -> class probablitites), use *two* `nn.Linear` objects. *Each of these are called a layer in the network*
* The first layer (aka the hidden layer) will transfrom the input matric of shape `batch_size x 784` into an intermediate output matrix shape `batch_size x hidden_size`, where `hidden_size` is a preconfigured parameter (ex. 32 or 64)
* The intermediate outputs are then passed into a non-linear *activation function*, which operated on individual elements of the output matrix
* The result of the activation function (size of `batch_size x hidden_size`) is passed into the second layer (aka the output layer), which transforms it into a matrix of size `batch_size x 10`, identical to the output of the logistic regression model

### Step by step example
Starting with a batch of input tensors. First flatten the `1x28x28` images into vectors of size `784`, so they can be passed into an `nn.Linear` object.
```
for images, labels in train_loader:
    inputs = images.reshape(-1, 784)
    break
```
Create the *hidden layer* by creating a `nn.Linear` object. Set the size of the output from the hidden layer to `32`. This number can be increased or decreased to change the learning capacity of the model.
```
input_size = inputs.shape[-1]
hidden_size = 32
layer1 = nn.Linear(input_size, hidden_size)
```
Now pass `inputs` through `layer1` to compute the intermediate outputs.
```
layer1_outputs = layer1(inputs)
```
The image vectors of size `784` are transformed into intermediate output vectors of length `32` by performing a matrix multiplication of `inputs` matrix with the transposed weights matrix of `layer1` and adding the bias. 
```
layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias
```
So `torch.Size` is now `[128, 32]`

Thus, `layer1_outputs` and `inputs` have a linear relationship. Each element of `layer_outputs` is a weighted sum of elements from `inputs`. So, even while training the model and modifying the weights, `layer1` can only capture linear relationships between `inputs` and `outputs`.

![layers](https://i.imgur.com/inXsLuq.png)

Next is the activation function for the outputs. The *Rectified Linear Unit (RelU) function* can be used with their formula `relu(x) = max(0, x)` which simply replaces negative values in a given tensor with the value 0. ReLU is a non-linear function, as seen here visually:
![relu](https://i.imgur.com/yijV4xF.png)

Use the `F.relu` method to apply ReLU to the elements of a tensor.

Ex:
```
F.relu(torch.tensor([[1, -1, 0], 
                     [-0.1, .2, 3]]))
```
which transforms into
```
tensor([[1.0000, 0.0000, 0.0000],
        [0.0000, 0.2000, 3.0000]])
```

Now apply the activation function to `layer1_outputs`
```
relu_outputs = F.relu(layer1_outputs)
```


