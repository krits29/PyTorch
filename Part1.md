# What is PyTorch?

PyTorch is a library for processing tensors. 
**Tensors** are numbers, vecotrs, matrices, any n-dimensional array

```
t1 = torch.tensor(0.4)
t1.dtype

t2 = torch.tensor([1, 2, 3, 4])

t3 = torch.tensor([5, 6], [7, 8], [9, 10])
```

Use `.shape` to return the length of each dimension
```
t1.shape
#returns torch.Size([])

t2.shape
#returns torch.Size([4])

t3.shape
#returns torch.Size([3, 2])
```
Normal arithmetic operations can be used with tensors.

# Gradients
A gradient is just the derivative. With PyTorch, the derivative can be automatically computed using a feature called *autograd*.

When defining a tensor, an additional parameter can be added to represent whether or not the tensor can have a gradient. If true, the derivative can be computed with respect to the tensor.

```
x = torch.tensor(3) #basic
y = torch.tensor(4, requires_grad=True) #another parameter
z = torch.tensor(5, requires_grad=True) #another parameter
```

Use the `.backward` method to compute the derivative

```
a = x * y + z
a.backward()
```
Notice how the variable `a` has three different tensor inputs. So now, when the dervative is computed, it must be taken with respect to each tensor input.

To display the derivatives of `a` with respect to the input tensors, the `.grad` property can be used.

```
#da/dx
x.grad

#da/dy
y.grad

#da/dz
z.grad
```
Since `x` did not have `requires_grad` set to `true`, the gradient would be `None`.

For `y`, the gradient of `a` with respect to `y` would be `tensor(3)`. 

And for `z`, the gradient of `a` with respect to `z` would be `tensor(1)`.

# Numpy
Numpy is an open source library using in Python for mathematical and scientific computing. It's also known for efficiency and optimized operations on large arrays with multiple dimensions. Numpy also supports a bunch of libraries (ex: Pandas, Matplotlib, OpenCV).

PyTorch incorporates Numpy along with all the other libraries it uses.

Make sure to import Numpy by saying `import numpy as np` at the top of the code.

An array in Numpy can be converted into a PyTorch tensor by using `torch.from_numpy` and vice verca using the `.numpy` method.
```
import numpy as np

#create a numpy array
x = np.array([1, 2], [3, 4])

#numpy array -> torch tensor
y = torch.from_numpy(x)

#torch tensor -> numpy array
z = y.numpy()
```
Most datasets will be read and processed as Numpy arrays.

