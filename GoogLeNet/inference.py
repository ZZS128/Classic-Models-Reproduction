import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import GoogLeNet, Inception
from PIL import Image



num2label = {
    0: 'cat',
    1: 'dog'
    }

pic_root = 'GoogLeNet/new_data/1.png'

# 模型推理函数
def inference(model, pic_root):
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    pic_transform = transforms.Compose([transforms.Resize(224, 224),transforms.ToTensor(), normalize])
    test_x = pic_transform(Image.open(pic_root))
    test_x = test_x.unsqueeze(0)  # 增加batch维度
    
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
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('./GoogLeNet/googlenet.pth'))
    pred_lab = inference(model, pic_root)
    print(f"这个应该是: {num2label[pred_lab]}")