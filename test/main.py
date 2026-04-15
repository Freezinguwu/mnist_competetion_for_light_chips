import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
import os
import numpy as np
import entrance
from time import time
from sys import argv
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Conv2d
from torch import Tensor
from torch.utils.data import Subset

def get_data_loader(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    # 下载 MNIST 数据集，并应用转换
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # ✅ 限制数据量（比如只用1000条）----tips：调试用，正式用把这个可以删掉
    subset_size = 10000
    train_dataset = Subset(train_dataset, range(subset_size))
    # test_dataset = Subset(test_dataset, range(300))  # 测试集可以更小
    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

#演示部分
def data_display():
    batch_size = 1
    num_workers = 4
    train_loader, test_loader = get_data_loader(batch_size, num_workers)

    # 获取第一个 batch
    images, labels = next(iter(train_loader)) 
    first_img = images[0]  # 取出第一个图像
    images_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    first_img_224 = images_224[0]  # 取出第一个图像的细分版本
    print(first_img.shape)
    # --- 开始画图展示 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # 左图：原始的 28x28 (像素化的数字图像)
    axes[0].imshow(first_img.permute(1, 2, 0).numpy(), cmap='gray')
    axes[0].set_title("Original: 28x28\n(Pixelated MNIST)")
    # 右图：细分后的 224x224 (边缘变平滑了，像一张真正的照片)
    # 绘图需要把 [C, H, W] 转回 [H, W, C]
    axes[1].imshow(first_img_224.permute(1, 2, 0).numpy())
    axes[1].set_title("Resized: 224x224\n(Ready for ResNet50)")
    plt.show()

if __name__ == "__main__":
    data_display()
    # 1. 加载模型
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 10)
    model.cuda()
    # 3. 移动到 GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 100  # 初始化batch批次(tips:报错就是显存太大，减一点size)
    num_workers = 4   # GPU工作的线程 
    train_loader, test_loader = get_data_loader(batch_size, num_workers)
    print("开始训练.........")
    start = time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 1. 搬运到 GPU
        images, labels = images.cuda(), labels.cuda()
        images_32 = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
        # 2. 这里可以把 images_224 输入到 ResNet50 模型中，得到输出 logits
        output = model(images_32)
        # 3. 这里可以计算 Loss、跑 backward (如果你想训练的话)
        print(f"正在处理第 {batch_idx + 1} 个批次...")
        # 计算准确率和损失
        loss = criterion(output, labels)
        pred = output.argmax(dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / batch_size
        print(f"EPOCH ACC: {acc}")
        # 使用adam方法减少损失
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 训练继续
    print("开始测试.......")
    # 4. 测试阶段可以在 test_loader 上进行评估，计算准确率等指标
    end = time()
    print("time cost: ", end - start)
    start = time()
    correct_tmp = 0
    total = 0
    with torch.no_grad(): # 关键：关闭梯度计算，省显存！
        for batch_idx, (images, labels) in enumerate(test_loader):
          # 1. 搬运到 GPU
          images, labels = images.cuda(), labels.cuda()
          images_32_TEST = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
          # 2. 把 images_224 输入到 ResNet50 模型中，得到输出 logits
          output = model(images_32_TEST)
          print(f"正在处理第 {batch_idx} 个测试批次...")  
          # 3. 计算总准确率
          pred = output.argmax(dim=1) 
          correct = (pred == labels).sum().item() 
          correct_tmp += correct
          total += labels.size(0)
          acc = correct_tmp / total
          print(f"total acc: {acc}")
        print(f"测试结束.....")
    end = time()
    print("time cost: ", end - start)

    # 在训练完成后（测试之前或之后）添加
    torch.save(model.state_dict(), 'resnet18_mnist.pth')
    print("模型已保存为 resnet18_mnist.pth")



