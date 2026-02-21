import torch
import torch.utils.data as Data
from torchvision import transforms
from model import GoogLeNet, Inception
from torchvision.datasets import ImageFolder

num2label = {
    0: 'cat',
    1: 'dog'
    }

# 数据预处理，加载数据
def data_process():
    dara_root = 'GoogLeNet/data/test'
    # 数据集计算得出的均值和标准差
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    test_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),normalize])
    
    dataset = ImageFolder(root=dara_root, transform=test_transform)
    test_data_loader = Data.DataLoader(dataset, 
                                        batch_size=64, 
                                        shuffle=False, 
                                        num_workers=4)
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
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('./GoogLeNet/googlenet.pth'))
    test_data_loader = data_process()
    test(model, test_data_loader)

