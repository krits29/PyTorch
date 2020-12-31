# What is PyTorch?

PyTorch is a library for processing tensors. 
**Tensors** are numbers, vecotrs, matrices, any n-dimensional array

```
t1 = torch.tensor(0.4)
t1.dtype

t2 = torch.tensor([1, 2, 3, 4])

t3 = torch.tensor([5, 6], [7, 8], [9, 10])


#use .shape to return the length of each dimension

t1.shape
#returns torch.Size([])

t2.shape
#returns torch.Size([4])

t3.shape
#returns torch.Size([3, 2])

#can use normal arithmetic operations with tensors

```

