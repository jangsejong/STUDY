import warnings
warnings.filterwarnings('ignore')

import os, datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm

import FinanceDataReader as fdr


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 0
seed_everything(seed)

#1. data

stock_list = pd.read_csv('data/stock_list.csv')
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))
stock_list = stock_list.sort_values(by=['종목코드'])
stock_list

start_date = '2018-01-01'
end_date = '2021-12-03'   # 예측하고 싶은 week의 마지막 날짜
num_val = 2

features = ['Close']    # ['Close', 'Open', 'High', 'Low', 'Volume', 'Change']
norm_factors = {'Close': 1e6}

seq_len = 8   # 8주의 데이터로 다음 주의 종가 예측
dim_f = len(features)
dim_d = 5   # number of business days

def get_data(code, start_date, end_date):
    stock_data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
    
    # 토요일, 일요일 제거
    week_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns = ['Date'])
    stock_data = pd.merge(week_days, stock_data.drop(columns=['Change']), how = 'left')   
    
    # 주말 외 휴일의 NaN 값을 이전 날의 데이터로 매꿈.
    # 상장폐지 종목들의 폐지 날짜 이후의 종가가 NaN이 아닌 마지막 종가로 대체가 되어 
    # 학습 데이터에 포함되게 되지만, 성능에 큰 영향은 없었음.
    stock_data = stock_data.ffill()  
    
    return stock_data

def preprocess(df):
    df = df[features]
    df.dropna(how='any', axis=0, inplace=True)
    df = df[(len(df)%5):]
    for column in df.columns:
        df[column] /= norm_factors[column]
    return df.values.reshape(-1,dim_d,dim_f)    # shape = [num_weeks, num_business_days, num_features]


def split(data):
    train = data[:-(num_val+1)]
    val = data[-(seq_len+num_val):] if num_val>0 else []
    test = data[-(seq_len+1):]
    return train, val, test
        
    
def to_xy(time_series):
    xy = []
    for i in range(seq_len, len(time_series)):
        x = time_series[i-seq_len:i]
        y = time_series[i,:,features.index('Close')]
        xy.append({'x': x, 'y': y})
    return xy

train_data, val_data, test_data = [], [], []
for code in tqdm(stock_list['종목코드'].values):
    df = get_data(code, start_date, end_date)
    train, val, test = split(preprocess(df))
    train_data += to_xy(train)
    val_data += to_xy(val)
    test_data += to_xy(test)
    
len(train_data), len(val_data)

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        x = torch.tensor(self.data[i]['x'], dtype=torch.float32)
        y = torch.tensor(self.data[i]['y'], dtype=torch.float32)
        return x, y
    
trainset = StockDataset(train_data)
valset = StockDataset(val_data)


# 2. model

# day가 아닌 week 단위로 예측.
# input sequence의 각 element는 한 주의 모든 feature들을 flatten해서 만듬. 

class StockPredictor(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        
        self.seq_encoder = nn.LSTM(
            input_size=dim_d*dim_f,  
            hidden_size=h, 
            num_layers=n, 
            bidirectional=False, 
            batch_first=True
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(h, dim_d)
        )
        
    def forward(self, x):
        x = x.reshape(*x.shape[:2], -1)
        
        x = self.seq_encoder(x)[0]
        x = x[:,-1]
        
        return self.linear(x)
    
#3. traininig

device = torch.device("cuda:0")

num_epochs = 1000
num_workers = 0

batch_size = 256
learning_rate = 5e-5

n = 2
h = 8

model_name = f'LSTM(n{n}h{h})'
model_path = f'models/{model_name}'
if not os.path.isdir(model_path):
    os.mkdir(model_path)

save_path = f'{model_path}/bs{batch_size}_lr{learning_rate}_seed{seed}_seq{seq_len}_date({start_date}~{end_date}).pt'

model = StockPredictor(n, h).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
if num_val > 0:
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

def step(batch, training):
    x = batch[0].to(device)
    y = batch[1].to(device)
    if training:
        model.train()
        output = model(x)
        loss = nn.L1Loss()(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        model.eval()
        with torch.no_grad():
            output = model(x)
            loss = nn.L1Loss()(output, y)
    return loss.item() * x.shape[0]


def run_epoch(loader, training):
    total_loss = 0
    for batch in tqdm(loader):
        batch_loss = step(batch, training)
        total_loss += batch_loss
    return total_loss/len(loader.dataset)


def show_loss_plot(train, val):
    print('train_loss', train[-1])
    print('val_loss  ', val[-1])
    plt.ylim(0, 0.005)
    plt.plot(train_loss_plot, label='train_loss')
    plt.plot(val_loss_plot, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MAE)')
    plt.legend()
    plt.show()

train_loss_plot, val_loss_plot = [], []
for epoch in range(num_epochs):
    train_epoch_loss = run_epoch(train_loader, training=True)
    
    val_epoch_loss = run_epoch(val_loader, training=False) if num_val>0 else 0
    
    ipd.clear_output(wait=True)

    print('epoch' , epoch)
    train_loss_plot.append(train_epoch_loss)
    val_loss_plot.append(val_epoch_loss)
    show_loss_plot(train_loss_plot, val_loss_plot)
    
    if min(val_loss_plot)==val_epoch_loss:
        torch.save(model.state_dict(), save_path)


# 4. test

model.eval()
model.load_state_dict(torch.load(save_path, map_location=device))

private_x = np.stack([xy['x'] for xy in test_data])
private_y = np.stack([xy['y'] for xy in test_data])
private_x.shape, private_y.shape

x = torch.tensor(private_x, dtype=torch.float32)
with torch.no_grad():
    pred = model(x.to(device))
    
pred = pred.detach().cpu().numpy() * norm_factors['Close'] 
pred.shape

def submission_df(pred, template):
    submission = template
    for code_id, p in enumerate(pred):
        code = stock_list.reset_index().loc[code_id,'종목코드']
        submission.loc[:,code] = list(np.zeros_like(p)) + list(p)
    columns = list(submission.columns[1:])
    submission.columns = ['Day'] + [str(x).zfill(6) for x in columns]
    return submission

sample_submission = pd.read_csv('sample_submission.csv')
submission_df(pred, sample_submission)


# data/stock_list.csv








