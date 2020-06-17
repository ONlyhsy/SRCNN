from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):#用RGB色彩空间来做，输入为3个通道，不要池化层了，最后分辨率应该是一样的
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64*3, kernel_size=9, padding=9 // 2)#(3,256,256)-->(64*3,256,256)
        self.conv2 = nn.Conv2d(64*3, 32*3, kernel_size=5, padding=5 // 2)#(64*3,256,256)-->(32*3,256,256)
        self.conv3 = nn.Conv2d(32*3, num_channels, kernel_size=5, padding=5 // 2)#(32*3,256,256)-->(3,256,256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x