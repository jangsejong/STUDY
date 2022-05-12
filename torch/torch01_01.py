import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(1)

print(x,y)


model = nn.Linear(1,1)  #인풋, 아웃풋


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# model.fit(x, y, epochs=100, batch_size=1)
def train(model, criterion, optimizer, x, y, batch_size=1, epochs=1):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
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

result = model(torch.FloatTensor([[4]]))
print('result: ', result)