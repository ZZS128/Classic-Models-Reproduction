import os
import random
from shutil import copy

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

split_rate = 0.1
file_path = 'ResNet18/Rice_Image_Dataset'
class_name = [cla for cla in os.listdir(file_path)]

# 划分训练集
mkfile('ResNet18/data/train')
for cla in class_name:
    mkfile('ResNet18/data/train/' + cla)

# 划分测试集
mkfile('ResNet18/data/test')
for cla in class_name:
    mkfile('ResNet18/data/test/' + cla)

for cla in class_name:
    cla_path = file_path + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'ResNet18/data/test/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'ResNet18/data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()