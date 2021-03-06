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
The first dimension tracks color channels. The second and third dimensions represents the height and width of the image in pixels, respectively. Since images in the MNIST dataset are grayscale, there's just one channel. Other datasets have images with color, in which case there are three channels: red, green, and blue (RGB).

Here are some sample values inside the tensor `print(img_tensor[0,10:15,10:15])`
```
tensor([[0.0039, 0.6039, 0.9922, 0.3529, 0.0000],
        [0.0000, 0.5451, 0.9922, 0.7451, 0.0078],
        [0.0000, 0.0431, 0.7451, 0.9922, 0.2745],
        [0.0000, 0.0000, 0.1373, 0.9451, 0.8824],
        [0.0000, 0.0000, 0.0000, 0.3176, 0.9412]])
```
The values range from 0 to 1, with 0 representing black, 1 white, and the values in between different shades of grey. `print(torch.max(img_tensor), torch.min(img_tensor))`
```
tensor(1.) tensor(0.)
```
Using `plt.imshow`, the tensor can be plotted as an image:
```
plt.imshow(img_tensor[0,10:15,10:15], cmap='gray');
```
![imshow](https://lh3.googleusercontent.com/fze6zrNtkU4KK79A4gOeOxP5PlW4CiKirOJjmj2Ezw3SDYnxhXHYM5-x8hjmxxbYp8Kl=s85)

### Training sets and Validation sets
When building real-world machine learning models, it is common to split the dataset into three parts:
1. **Training set** - used to train the model (ex: compute the loss and adjust the model's weights using gradient descent)
2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc), and pick the best version of the model
3. **Test set** - used to compare different models or approaches and report the model's final accuracy

In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images.


Since there's no predefined validation set, the 60,000 images must be split into training and validation datasets. 10,000 randomly chosen images can be set aside for validation by using the `random_split` method from PyTorch. The validation set should be created with a random sample since the training data is often sorted by the target labels.
```
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000])
```

Next, the data is split into batches. This is done by creating data loaders. The batch size is set for 128:
```
from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
```
In order to ensure that the data loader generates different batches in each eopch, `shuffle` is set to true. This helps with randomization and will generalze and speed up the training process. Since the validation data loader is only for evaluating the model, there is no need to shuffle the images.

### Model
Now that the data loaders are prepared, the model can be defined:
- A **logistic regression** model is almost identical to a linear regression model. It contains weights and bias matrices, and the output is obtained using simple matrix operations (`pred = x @ w.t() + b`)
- Use `nn.Linear` to create the model instead of manually creating and initializing the matrices
- Since `nn.Linear` expects each training example to be a vector, each `1x28x28` image tensor is *flattened* into a vector of size 784 `(28*28)` before being passed into the model
- The output for each image is a vector of size 10, with each element signifying the probability of a particular target label (ex: 0 to 9). The predicted label for an image is simply the one with the highest probability

This model is a lot larger with a large amount of parameters
```
import torch.nn as nn

input_size = 28*28
num_classes = 10
```
Logistic Regression Model
```
model = nn.Linear(input_size, num_classes)
```
Weights `model.weight`
```
Parameter containing:
tensor([[ 0.0009, -0.0116, -0.0353,  ...,  0.0250,  0.0174, -0.0170],
        [ 0.0273, -0.0075, -0.0141,  ..., -0.0279,  0.0321,  0.0207],
        [ 0.0115,  0.0028,  0.0332,  ...,  0.0286, -0.0246, -0.0237],
        ...,
        [-0.0151,  0.0339,  0.0293,  ...,  0.0080, -0.0065,  0.0281],
        [-0.0011,  0.0064,  0.0177,  ..., -0.0050,  0.0324, -0.0150],
        [ 0.0147, -0.0001, -0.0042,  ..., -0.0102,  0.0343, -0.0263]],
       requires_grad=True)
```
Biases `model.bias`
```
Parameter containing:
tensor([ 0.0080,  0.0105, -0.0150, -0.0245,  0.0057, -0.0085,  0.0240,  0.0297,
         0.0087,  0.0296], requires_grad=True)
```
