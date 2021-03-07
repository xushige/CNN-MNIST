import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader

def get_trans():
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])
    return trans

train_data = torchvision.datasets.MNIST(root="mnist", train=True, download=False, transform=get_trans())
test_data = torchvision.datasets.MNIST(root="mnist", train=False, download=False, transform=get_trans())

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 6, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.flc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.flc(x)
        return out

cnn = CNN().cuda()
EP = 20
B_S = 128

train_loader = DataLoader(train_data, batch_size=B_S, shuffle=True)
test_loader = DataLoader(test_data, batch_size=B_S*10, shuffle=True)

LR = 5e-4
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for i in range(EP):
    correct_num = 0
    for img, label in train_loader:
        img, label = img.cuda(), label.cuda()
        out = cnn(img)
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for img,label in test_loader:
        img, label = img.cuda(), label.cuda()
        out = cnn(img)
        _, pre = torch.max(out, 1)
        correct_num += (pre == label).sum()
    acc = correct_num.cpu().numpy()/len(test_data)
    print("第%d次迭代结束，正确率为%f" % (i+1, acc))

save_path = os.path.join('cnn_model2.pth.tar')
torch.save({"state_dict":cnn.state_dict()}, save_path)
