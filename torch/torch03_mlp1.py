from turtle import shape
import numpy as np
from sklearn.linear_model import SGDClassifier 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE)
# torch : 1.9.0+cu111  사용DEVICE : cuda

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
y = np.array([11,12,13,14,15,16,17,18,19,20])
x = np.transpose(x)
x = torch.FloatTensor(x)#.to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1)#.to(DEVICE)

print(x,y)
print(x.shape,y.shape)


# model = nn.Linear(1,1).to(DEVICE)  #인풋, 아웃풋
model = nn.Sequential(
    nn.Linear(10,2),
    nn.Linear(2,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1),
)#.to(DEVICE)

criterion = nn.MSELoss() # MSELoss : mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.001)
# model.fit(x, y, epochs=100, batch_size=1)
def train(model, criterion, optimizer, x, y, batch_size=1, epochs=1):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss(hypothesis, y) # 에러
    # loss = nn.MSELoss()(hypothesis, y) 
    loss = F.mse_loss(hypothesis, y) 
    loss.backward()
    optimizer.step()
    return loss.item() #loss.item()은 loss의 실수값을 반환한다.

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('Epoch: {}, Loss: {}'.format(epoch, loss))

#4. 평가, 예측
# loss = model.evaluate(x, y)    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss: ', loss2)

result = model(torch.Tensor([[10, 1.3]]))
print('result :', result.item())