import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 6, 1, 2),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.flc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.flc(x)
        return out


cnn = CNN().cuda().eval()
load = torch.load('cnn_model_drop.pth.tar')
cnn.load_state_dict(load['state_dict'])

for i in range(10):
    img = '%d.jpg' % (i)
    img = Image.open(img).resize((28, 28))
    img = img.convert("L")

    img = np.array(img)
    img = torch.from_numpy(img).float()
    img = img.view(1, 1, 28, 28).cuda()

    out = cnn(img)
    _, pre = torch.max(out, 1)
    print("数字为{%d}" % (pre))