import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as d
import os
import tool as t
import pandas as pd

X_train = torch.from_numpy(np.load('npy/train/X_train.npy'))
X_test = torch.from_numpy(np.load('npy/test/X_test.npy'))
Y_train = torch.from_numpy(np.load('npy/train/Y_train.npy'))
Y_test = torch.from_numpy(np.load('npy/test/Y_test.npy'))

train_data = d.TensorDataset(X_train, Y_train)
test_data = d.TensorDataset(X_test, Y_test)

layer_list = [64*42, 128*13, 256*4, 256*4]

class block(nn.Module):
    def __init__(self, inchannal, outchannal, stride, layer_num):
        super(block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inchannal, outchannal, 6, stride, 1),
            nn.BatchNorm1d(outchannal),
            nn.ReLU()
        )
        self.block_conv = nn.Conv1d(outchannal, outchannal, 3, 1, 1)
        self.block_flc = nn.Linear(layer_list[layer_num], 5)
        self.optimizer = torch.optim.Adam(self.parameters(), 5e-4)
    def forward(self, x, y, istrain):
        y_matrix = nn.functional.one_hot(y, 5).float()
        out = self.conv(x)
        mse_input = self.block_conv(out)
        mse_input = mse_input.view(mse_input.size(0), -1)
        mse_input = self.block_flc(mse_input)
        mseloss = 0.99*nn.MSELoss()(mse_input, y_matrix)
        ce_input = out.view(out.size(0), -1)
        ce_input = self.block_flc(ce_input)
        celoss = 0.01*nn.CrossEntropyLoss()(ce_input, y)
        total_loss = mseloss + celoss

        if istrain:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return out.detach()
class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()
        self.layer0 = block(3, 64, 3, 0)
        self.layer1 = block(64, 128, 3, 1)
        self.layer2 = block(128, 256, 3, 2)
        self.flc = nn.Linear(layer_list[-1], 5)
    def forward(self, x, y, istrain):
        x = self.layer0(x, y, istrain)
        x = self.layer1(x, y, istrain)
        x = self.layer2(x, y, istrain)
        x = x.view(x.size(0), -1)
        out = self.flc(x)
        return out

net = convnet().cuda()
EP = 40

B_S = 128

train_loader = d.DataLoader(train_data, batch_size=B_S, shuffle=True)
test_loader = d.DataLoader(test_data, batch_size=B_S, shuffle=True)


optimizer = torch.optim.Adam(net.flc.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

view_list = []
for i in range(EP):
    cor = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        net.train()
        out = net(data, label, True)
        loss = loss_fn(out, label)
        view_list.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for data, label in test_loader:
        data, label = data.cuda(), label.cuda()
        net.eval()
        out = net(data, label, False)
        _, pre = torch.max(out, 1)
        cor += (pre == label).sum()
    acc = cor.cpu().numpy()/len(test_data)
    print("第%d次迭代结束，loss为%f,正确率为%f" % (i+1, loss, acc))

plt.plot(view_list)
plt.show()
