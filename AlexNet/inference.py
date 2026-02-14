import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import AlexNet
import random

num2label = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat', 
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
    }

# 生成随机测试样本的函数
def generate_random_sample():
    test_dataset = FashionMNIST(
        root='./AlexNet/Dataset',
        train=False,
        transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
        download=True)
    
    random_index = random.randint(0, len(test_dataset) - 1)
    test_x, test_y = test_dataset[random_index]
    test_x = test_x.unsqueeze(0)  # 增加batch维度
    return test_x, test_y

# 模型推理函数
def inference(model, test_x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        test_x = test_x.to(device)
        output = model(test_x)
        pred_lab = torch.argmax(output, dim=1)
        pred_lab = pred_lab.cpu().item()
    return pred_lab

if __name__ == "__main__":
    model = AlexNet()
    model.load_state_dict(torch.load('./AlexNet/alexnet.pth'))
    for i in range(10):
        test_x, test_y = generate_random_sample()
        pred_lab = inference(model, test_x)
        print(f"真实标签: {num2label[test_y]}, 预测标签: {num2label[pred_lab]}")