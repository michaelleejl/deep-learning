import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)  # weights: 5x3 instances of standard normal distribution
b = torch.randn(3, requires_grad=True)  # biases: 3 instances of standard normal distribution
# requires_grad: needed for gradient descent, off by default
# do w.requires_grad(True) to set it after initialisation

# chain together the inputs and create a directed acyclic graph of Function objects
# the leaves are the inputs, the roots are the outputs
# graph is dynamically reconstructed after every backwards pass, so can use
# control flow to dynamically update shape, size, and operations after every backwards pass
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# functions applied to compute computational graph is an object of class Function
# objects of class Function have a reference to a back propagation function in grad_fn
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# compute partial derivatives 𝛿loss/𝛿w and 𝛿loss/𝛿b
loss.backward()
print(w.grad)
print(b.grad)

# Note 1: The grad field only exists on leaf nodes with the requires_grad set to true
# Note 2: Gradient calculations can only be performed once on a graph.
#         If we need several gradient calls, do loss.backward(retain_graph=True)


# TURNING OFF GRADIENT TRACKING
# Method 1
with torch.no_grad():
    z1 = torch.matmul(x, w) + b

# Method 2
z2 = torch.matmul(x, w) + b
z2_det = z2.detach()

# Why you might want to turn off gradient tracking
# Reason 1: Mark parameters as frozen
# Reason 2: Optimize the forward pass (for example if you've already trained the nn)


# THE JACOBIAN
# The code above is useful for scalar outputs / losses
# If you have an output / loss which is not a scalar, but an arbitrary vector [z_0, ..., z_n]
# Let's say you want to compute 𝛿loss/𝛿w, this is the sum over z_i of 𝛿error/𝛿z_i * 𝛿z_i/𝛿w
# The Jacobian, J, has the property that J[i][j] = 𝛿z_i/𝛿w_j
# So [𝛿loss/𝛿w_0, 𝛿loss/𝛿w_1, ..., 𝛿loss/𝛿w_m]^T = [𝛿loss/𝛿z_0, ..., 𝛿loss/𝛿z_n]^T * J.
# If you call backward(v), it computes v^T * J
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"Second call\n{inp.grad}")  # Note gradients accumulate, get added to the grad field of Function
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
