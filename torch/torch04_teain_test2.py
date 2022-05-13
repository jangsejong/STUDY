from pickletools import optimize
from re import X
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE)

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
print(x_train, x_train.shape)

#2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 4),
    nn.Linear(4, 1),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, criterion, optimizer, x_train, y_train, batch_size=1):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 100
for epoch in range (1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('Epoch : ', epoch, 'Loss : ', loss)
    
#4. 평가, 예측
# loss = model.evaluate(x, y)    
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        predict = model(x_test)
        loss2 = criterion(predict, y_test)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss: ', loss2)

result = model(x_test.to(DEVICE))
# print('result: ', result)
# print('result: ', result.cpu().data.numpy())
print('result: ', result.cpu().detach().numpy())


# tensor를 numpy로 변환

 