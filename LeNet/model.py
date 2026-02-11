import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.con3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lin5 = nn.Linear(in_features=16*5*5, out_features=120)
        self.lin6 = nn.Linear(in_features=120, out_features=84)
        self.lin7 = nn.Linear(in_features=84, out_features=10)
        self.sig = nn.Sigmoid()
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.sig(self.con1(x))
        x = self.pool2(x)
        x = self.sig(self.con3(x))
        x = self.pool4(x)
        x = self.flat(x)
        x = self.lin5(x)
        x = self.lin6(x)
        x = self.lin7(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))
