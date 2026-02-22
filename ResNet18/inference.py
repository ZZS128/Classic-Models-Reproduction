import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import ResNet18, Residual
from PIL import Image



num2label = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
    }

pic_root = 'ResNet18/new_data/0.jpg'

# 模型推理函数
def inference(model, pic_root):
    normalize = transforms.Normalize([0.0420662, 0.04281093, 0.04413987], [0.03315472, 0.03433457, 0.03628447])
    pic_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    test_x = pic_transform(Image.open(pic_root).convert('RGB'))
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
    model = ResNet18(Residual)
    model.load_state_dict(torch.load('ResNet18/resnet18.pth'))
    pred_lab = inference(model, pic_root)
    print(f"这个应该是: {num2label[pred_lab]}")