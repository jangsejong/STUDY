from pickletools import optimize
import datasets
from sklearn.datasets import load_breast_cancer, load_iris
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE)

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)

x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)

x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape) # (105, 4)
print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
print(type(x_train), type(y_train))

#2. 모델구성
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    # nn.Sigmoid(),
).to(DEVICE)

#3. 컴파일, 훈련

# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train, batch_size=1):
    # model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 1000
for epoch in range (1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('Epoch : {:4d} / {:4d}, Loss : {:.8f}'.format(epoch, EPOCHS, loss))
    
# #4. 평가, 예측
# # loss = model.evaluate(x, y)    
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        predict = model(x_test)
        loss2 = criterion(predict, y_test)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss: ', loss2)

# y_predict = (model(x_test) > 0.5).float()
y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict)

scores = (y_predict == y_test).float().mean()
print('정확도: ', scores)

from sklearn.metrics import accuracy_score
scores2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('정확도: ', scores2)

scores3 = accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())
print('정확도: ', scores2)

'''
최종 loss:  0.923826277256012
tensor([1, 2, 1, 0, 1, 1, 0, 0, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2,
        1, 2, 0, 1, 2, 2, 0, 0, 2, 0, 1, 0, 1, 0, 2, 1, 0, 2, 2, 1, 1],
       device='cuda:0')
정확도:  tensor(0.9111, device='cuda:0')
정확도:  0.9111111111111111
정확도:  0.9111111111111111
'''