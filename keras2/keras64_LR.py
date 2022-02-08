import numpy as np


x = 0.5
y = 0.8
w = 0.5
lr = 0.1
epochs = 300

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2
    
    print("Loss :" , round(loss,4), "\tPrediction :" , round(predict,4))
    
    up_predict = x * (w + lr)
    up_loss = (up_predict - y) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (down_predict - y) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
        
    print("loss :" , round(loss,4), "\tPrediction :" , round(predict,4))
