import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import VGG16


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

# 数据预处理，加载数据
def data_process():
    test_dataset = FashionMNIST(
        root='./VGG16/Dataset',
        train=False,
        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
        download=True)

    test_data_loader = data.DataLoader(test_dataset, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        num_workers=0)
    return test_data_loader

# 测试函数
def test(model, test_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_num = 0

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for test_x, test_y in test_data_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            output = model(test_x)
            pred_lab = torch.argmax(output, dim=1)

            test_acc += torch.sum(pred_lab == test_y.data)
            test_num += test_x.size(0)
            if test_num % 1000 == 0:
                print(f"已测试{test_num}个样本")
    test_accuracy = test_acc.double().item() / test_num
    print("测试集上的准确率为：", test_accuracy)



if __name__ == "__main__":
    model = VGG16()
    model.load_state_dict(torch.load('./VGG16/vgg16.pth'))
    test_data_loader = data_process()
    test(model, test_data_loader)

