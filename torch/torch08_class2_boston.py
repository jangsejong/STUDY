from pickletools import optimize
import datasets
from sklearn.datasets import load_breast_cancer, load_boston
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE)

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape) # (398, 30) torch.Size([398])
print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
print(type(x_train), type(y_train))

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 16),
#     nn.ReLU(),
#     nn.Linear(16, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # super(Model, self).__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 16)
        self.l5 = nn.Linear(16, 16)
        self.l6 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        
    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(out1)
        out2 = self.relu(out2)
        out3 = self.l3(out2)
        out3 = self.relu(out3)
        out4 = self.l4(out3)
        out4 = self.relu(out4)
        out5 = self.l5(out4)
        out5 = self.relu(out5)
        out6 = self.l6(out5)
        out7 = self.relu(out6)
        return out7

model = Model(13, 1).to(DEVICE)


#3. 컴파일, 훈련

criterion = nn.MSELoss(reduction='sum') #reduction: 'mean' or 'sum'

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train, batch_size=1):
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
# print(y_predict)

y_predict = model(x_test)

# scores = (y_predict == y_test).float().mean()
# print('정확도: ', scores)

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# scores2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
# print('정확도: ', scores2)

# scores3 = accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())
# print('정확도: ', scores2)


r2_scores = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print('R2: {:.4f}'.format(r2_scores))

# # result = model(x_test.to(DEVICE))
# # # print('result: ', result)
# # # print('result: ', result.cpu().data.numpy())
# # print('result: ', result.cpu().detach().numpy())