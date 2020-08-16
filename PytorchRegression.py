import numpy as np
import matplotlib.pyplot as plt
import torch


xTrain = np.array([[4.7], [2.4], [7.5], [7.1], [4.3], [7.8],[8.9],[5.2],[4.59],[2.1],[8],[5],[7.5],[5],[4],[8],[5.2],[4.9],[3],[4.7],[4],[4.8],[3.5],[2.1],[4.1]],dtype=np.float32)
yTrain = np.array([[2.6], [1.6], [3.09], [2.4], [2.4], [3.3],[2.6],[1.96],[3.13],[1.76],[3.2],[2.1],[1.6],[2.5],[2.2],[2.75],[2.4],[1.8],[1],[2],[1.6],[2.4],[2.6],[1.5],[3.1]],dtype=np.float32)

XTrain = torch.from_numpy(xTrain)
YTrain = torch.from_numpy(yTrain)

plt.figure(figsize=(12,8))
plt.scatter(XTrain, YTrain, label='Original Data', s=250, c='g')
plt.legend()
#plt.show()

#training defaulted to not requires grad
#setup model parameters
inputSize = 1
hiddenSize = 1
outputSize = 1
learningRate = 1e-6

w1 = torch.rand(inputSize, hiddenSize, requires_grad=True)
w2 = torch.rand(hiddenSize, inputSize, requires_grad=True)

#the higher the index the better fit
for index in range(1, 10000):
    #forward pass to get predicted values first using random weights (w1 and w2)
    yPred = XTrain.mm(w1).mm(w2)
    #identify loss by comparing predicted values to actual values - standard error function for linear regression, sum of squared errors
    loss = (yPred - YTrain).pow(2).sum()

    if index % 50 == 0:
        print(index, loss.item())
    
    #backward pass to determine new gradient and tweak parameters
    loss.backward()
    with torch.no_grad():
        w1 -= learningRate * w1.grad
        w2 -= learningRate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

#print(w1, w2) finalized weights
#visualize fitted line compared to actual data
predicted = XTrain.mm(w1).mm(w2)
predicted = predicted.detach().numpy()
plt.plot(xTrain, predicted, label='Fitted Line')
plt.show()





