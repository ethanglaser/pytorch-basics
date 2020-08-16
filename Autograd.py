import torch

tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[7,8,9],[10,11,12]])

tensor1.requires_grad_() #when true, tracks computations for tensor in forward phase and calculates gradients in backward phase, default false
#.grad and .grad_fn initialized to None because no passes have been made yet

outputTensor = tensor1 * tensor2 #requires grad automatically true
#calculated tensor DOES have grad_fn
outputTensor = (tensor1 * tensor2).mean() #also has gradient function
outputTensor.backward() #initialized grad for tensor 1 (partial derivatives with respect to output) but not 2 because only 1 set requires_grad_()

newTensor = tensor1 * 3 #requires grad true, mulbackward

# differentiate between grad and no grad
def calculate(t):
    return t * 2
@torch.no_grad()
def calculateNoGrad(t):
    return t * 2
#if any input tensor has requires grad, output will as well
#.backward() calculates the gradients





