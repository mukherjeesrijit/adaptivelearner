# Problem

Based on your proficiency levels, I'll create an example and a coding problem tailored to help you learn PyTorch Coding for Deep Learning step by step.

**Example: Understanding PyTorch Tensors**

Let's start with understanding what PyTorch tensors are.
In this example, we will work with the basic building block of any deep learning model: the tensor. We will create a simple tensor and manipulate it to get a feel for how these objects work in PyTorch.

```python
import torch

# Create a 3x3 tensor filled with zeros
tensor = torch.zeros(3, 3)
print(tensor)

# Modify some values
tensor[0, 0] = 5
tensor[1, 2] = 7

# Print the modified tensor
print(tensor)
```

Output:
```python
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
tensor([[5., 0., 0.],
        [0., 0., 7.],
        [0., 0., 0.]])
```

In this example, we created a tensor with shape (3, 3) filled with zeros. We then modified some values at specific positions in the tensor.

**Problem: LeetCode Style - Tensors**

Now that you have a feel for PyTorch tensors, here's your first problem to challenge yourself:
```python
import torch

# Create two 2D tensors with shapes (3, 4) and (4, 5)
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4, 5)

# Perform matrix multiplication on tensor1 and tensor2
result = torch.matmul(tensor1, tensor2)

# What is the shape of the result?
print(result.shape)
```

**Challenge:** Fill in the blank to get the correct output.
Note: The expected output should be a comment.

**Instructions:**

* Complete the problem by writing the necessary code to perform matrix multiplication on `tensor1` and `tensor2`.
* Print the shape of the resulting tensor using `print(result.shape)`.
* Write the expected output as a comment in the format: `# Expected output: (__,__)`

**Note:** This is just one example, but there are many more concepts to explore in PyTorch!
