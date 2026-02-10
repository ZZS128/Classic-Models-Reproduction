import torch

print("Check PyTorch Version")
flag = torch.cuda.is_available()
print(f"CUDA Available: {flag}")

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(f"Using device: {device}")
print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(torch.rand(3,3).cuda())

print("Check Cuda&Cudnn Version")
print(torch.version.cuda)
print(torch.backends.cudnn.version())