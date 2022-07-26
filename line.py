import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

torch.manual_seed(1)
#data loading
data=pd.read_csv('line.csv')

def minmax(data):
    numerator=data-min(data)#상대비율
    denominator=max(data)-min(data)#구간길이
    return numerator/(denominator+1e-7)

#mid price computing
high=data['High'].values
low=data['Low'].values
mid=(high+low)/2
x_train=[]
y_train=[]
x_test=[]
y_test=[]
sequent_length=7
train_size=int(len(mid)*0.8)
mid=minmax(mid)

for i in range(0,train_size-7):
    datax = []
    for j in range(7):
        datax.append(mid[i+j])
    x_train.append(datax)
    y_train.append([mid[i+7]])

x_train = torch.FloatTensor(x_train)
y_train=torch.FloatTensor(y_train)
print(y_train)

for i in range(train_size,len(mid)-sequent_length):
    datax = []
    for j in range(7):
        datax.append(mid[i+j])
    x_test.append(datax)
    y_test.append([mid[i+7]])
x_test=torch.FloatTensor(x_test)
y_test=torch.FloatTensor(y_test)



class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(7,1)  #입력 차원->7, 출력 차원->1
    def forward(self,x):
        return self.linear(x)

model=LinearModel()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-5)
epoch_number=500000
for epoch in range(epoch_number):
    prediction=model(x_train)
    cost=F.mse_loss(prediction,y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100==0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch,epoch_number,cost.item()))
plt.plot(y_test)
print(model(x_test).data.numpy())
plt.plot(model(x_test).data.numpy())
plt.legend(['original','prediction'])
plt.show()