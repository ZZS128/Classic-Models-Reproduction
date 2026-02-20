import os
import random
from shutil import copy

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

split_rate = 0.1
file_path = 'GooLeNet/data_cat_dog'
class_name = [cla for cla in os.listdir(file_path)]

# 划分训练集
mkfile('GooLeNet/data/train')
for cla in class_name:
    mkfile('GooLeNet/data/train/' + cla)

# 划分测试集
mkfile('GooLeNet/data/test')
for cla in class_name:
    mkfile('GooLeNet/data/test/' + cla)

for cla in class_name:
    cla_path = file_path + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'GooLeNet/data/test/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'GooLeNet/data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()