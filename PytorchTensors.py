import torch
import numpy as np

#default type is float32, change with .set_default_dtype()
tensorArray = torch.Tensor([[1,2,3], [4,5,6]])

numElements = torch.numel(tensorArray)
#several different types and ways to initialize different arrays or as random values
#torch.full -> specify shape and number that will fill all values, can also do .ones, .zeros, .zeros_like, .eye (identity)
tensorZero = torch.zeros_like(tensorArray)

#identify nonzero values in tensor using .nonzero
i = torch.tensor([[0,1,1],[2,2,0]])
v = torch.tensor([3,4,5,],dtype=torch.float32)
tensorSparse = torch.sparse_coo_tensor(i, v, [2, 5])

tensorInitial = torch.rand(2,3)
tensorInitial.fill_(10) # sets all values to 10
tensorNew = tensorInitial.add(5) #add 5 to every value
tensorNew.sqrt_() #modifies all elements to match operation
tensorX = torch.linspace(start=0.1, end= 10.0, steps=15) #creates tensor of values evenly spaced from start to end
tensorChunk = torch.chunk(tensorX, 3, 0) #split apart
tensorCat = torch.cat((tensorChunk[0], tensorChunk[1], tensorChunk[2]), 0) #join back together
#.view, .unsqueeze, .transpose can be used to reshape tensors - share same memory so if original is modified then view will be too

tensor1 = torch.tensor([[10,8,30],[40,5,6],[12,2,100]])
tensorSorted, indicesSorted = torch.sort(tensor1) #sorts each row, indices show order of values based on indices
tensorFloat = torch.FloatTensor([-1.1,-2.2,-3.3])
absolute = torch.abs(tensorFloat) #finds absolute value of each entry
#elementwise operations: .add, .mul, .div, .dot, .mv (matrix * vector), .mm (matrix * matrix) etc

tensorClamp = torch.clamp(tensorFloat, min=-3, max=-2) #limit range of tensor values
maxIndex = torch.argmax(tensor1,1) #identifies index of largest value of each row
minVal = torch.argmin(tensor1) #identifies index of smallest value in entire matrix

#converting between pytorch tensor and numpy array
tensor2 = torch.rand(4,3)
numpyFromTensor = tensor2.numpy() #converts tensor to array - share memory (change one will change other)
array1 = np.array([[1,2,3], [10,20,30], [100, 200, 300])
tensorFromNumpy = torch.from_numpy(array1) #converts array to tensor, share memory, .as_tensor works too

'''
CUDA ops
torch.cuda.is_available()
torch.cuda.init()
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.memory_allocated()
cuda = torch.device('cuda')
tensor1 = torch.tensor([1,2], device=cuda) (creates tensor for cuda device)
tensor2 = tensorCPU.cuda() (converts CPU tensor to GPU tensor)
with torch.cuda.device(1): (changes device)
operations can't be performed on tensors from different devices
'''


