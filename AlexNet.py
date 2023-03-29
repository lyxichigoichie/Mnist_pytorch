from torch import nn
from torch.nn import functional as F

'''严格按照Alex Krizhevsky的论文“ImageNet Classification with Deep Convolutional Neural Networks"
定义AlexNet'''
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9216, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, 10)
        
    def forward(self, x):
        out_conv1 = F.relu(self.conv1(x))
        out_pool1 = self.maxpool1(out_conv1)
        out_conv2 = F.relu(self.conv2(out_pool1))
        out_pool2 = self.maxpool2(out_conv2)
        out_conv3 = F.relu(self.conv3(out_pool2))
        out_conv4 = F.relu(self.conv4(out_conv3))
        out_conv5 = F.relu(self.conv5(out_conv4))
        out_pool3 = self.maxpool3(out_conv5)
        
        flatten_x = self.flatten(out_pool3)
        out_linear1 = F.relu(self.linear1(flatten_x))
        out_dropout1 = self.dropout1(out_linear1)
        out_linear2 = F.relu(self.linear2(out_dropout1))
        out_dropout2 = F.relu(out_linear2)
        out_linear3 = F.relu(self.linear3(out_dropout2))
        return out_linear3