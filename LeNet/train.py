from torchvision.datasets import FashionMNIST
from torchvision import transforms

train_dataset = FashionMNIST(
    root='./LeNet/Dataset',
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)