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


from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

print(len(train_set), len(test_set)) # 354 152
# (tensor([-0.3550,  0.3810, -1.0714, -0.2815,  0.7585,  1.7561,  0.7023, -0.7661,
#         -0.5227, -0.8624, -2.4992,  0.3317, -0.7465], device='cuda:0'), tensor([43.1000], device='cuda:0'))
print(train_set[0])

train_loader = DataLoader(train_set, batch_size=36, shuffle=True)
test_loader = DataLoader(test_set, batch_size=36, shuffle=False)



# 2. 모델구성

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

def train(model, criterion, optimizer, train_loader):

    total_loss = 0
        
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
    
        hypothesis = model(x_batch)
    
        loss = criterion(hypothesis, y_batch)
    
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss# / len(train_loader)

EPOCHS = 1000
for epoch in range (1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('Epoch : {:4d} / {:4d}, Loss : {:.8f}'.format(epoch, EPOCHS, loss))
    
# #4. 평가, 예측
# # loss = model.evaluate(x, y)    
def evaluate(model, criterion, test_loader):
    model.eval()
    total_loss = 0

    for x_batch, y_batch in train_loader:        
        with torch.no_grad():
            predict = model(x_batch)
            loss = criterion(predict, y_batch)
            total_loss += loss.item()
    return total_loss #/ len(test_loader)



loss2 = evaluate(model, criterion, test_loader)
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