import os
import math
from osimulator.api import load_gazelle_model
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
from time import time
from torchvision.models import resnet50, ResNet50_Weights
from torch import Tensor
from torch.utils.data import Subset

# --- 1. 全局配置改为 CPU ---
device = torch.device("cpu")
input_type = 'uint4'

# 加载模拟器 (通常模拟器本身就是在 CPU 上运行的 C++ 库)
simulator_instance = load_gazelle_model()
simulator_instance.int2uint = True

def omac_matmul(inp, wgt):
    # CPU 版本不需要 .cpu()，直接转 numpy
    inp_np = inp.detach().numpy()
    wgt_np = wgt.detach().numpy()
    print(f"DEBUG: 正在执行模拟器矩阵乘法, 输入形状: {inp.shape}, 权重形状: {wgt.shape}")
    # 执行模拟器计算
    output_np = simulator_instance(inp_np, wgt_np, inputType=input_type)
    
    # 如果返回的是 numpy 数组，转回 torch tensor
    if isinstance(output_np, np.ndarray):
        return torch.from_numpy(output_np)
    return torch.from_numpy(output_np.numpy())

def get_data_loader(batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(        
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 限制数据量，CPU 跑 ResNet50 非常慢
    train_dataset = Subset(train_dataset, range(100)) # 进一步缩小到 100 条
    test_dataset = Subset(test_dataset, range(30))   
    
    # CPU 环境下 num_workers 建议设为 0，防止模拟器多线程冲突
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

class oMacConv2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def omac_forward(self, input: Tensor, weight: Tensor):
        # 这里的逻辑保持不变，但所有操作都在 CPU 上
        input_padded = F.pad(input, (self.padding[0],self.padding[0],self.padding[1],self.padding[1]))
        batch_size = input.shape[0]
        input_h, input_w = input.shape[2:4]
        
        output_h = (input_h + 2*self.padding[0] - (self.kernel_h + (self.kernel_h - 1) * (self.dilation[0] - 1)))//self.stride[0]+1
        output_w = (input_w + 2*self.padding[1] - (self.kernel_w + (self.kernel_w - 1) * (self.dilation[1] - 1)))//self.stride[1]+1

        input_vector = self.unfold(input).to(torch.int32).permute(0,2,1).contiguous()
        kernel_vector = torch.reshape(weight,[self.module.out_channels,-1]).to(torch.int32)
        input_vector = torch.reshape(input_vector,(-1,kernel_vector.shape[1])).contiguous()
        
        # 调用模拟器
        output = omac_matmul(input_vector.unsqueeze(0), kernel_vector.T.unsqueeze(0))
        
        # 重组形状
        output = torch.reshape(output, (batch_size, output_h, output_w, self.module.out_channels))
        output = output.permute(0,3,1,2).contiguous().to(torch.float32)
        return output
    
    @staticmethod
    def from_conv2d(cls, module):
        cls.module = module
        cls.kernel_h, cls.kernel_w = module.weight.shape[2:4]
        cls.padding = module.padding
        cls.dilation = module.dilation
        cls.stride = module.stride
        
        # 注意：Unfold 也要在 CPU 上运行
        cls.unfold = nn.Unfold(kernel_size=(cls.kernel_h, cls.kernel_w), 
                               stride=module.stride, padding=module.padding, dilation=module.dilation)
        
        # 简单量化权重到 int4
        wscale = 1.0 
        wqmin, wqmax = -8, 7
        w = torch.round(module.weight.data / wscale)
        cls.w = torch.clamp(w, wqmin, wqmax)
        return cls
        
    def forward(self, input: Tensor) -> Tensor:
        # 激活量化
        aqmin, aqmax = 0, 15
        inp = torch.clamp(torch.round(input), aqmin, aqmax)
        return self.omac_forward(inp, self.w)

def replace_conv2d(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_conv = oMacConv2d()
            oMacConv2d.from_conv2d(new_conv, child)
            setattr(model, name, new_conv)
        else:
            replace_conv2d(child)

if __name__ == "__main__":
    batch_size = 1 # CPU 运行 batch 不要设大
    train_loader, test_loader = get_data_loader(batch_size)
    
    # 加载 ResNet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048, 10)
    
    # 替换卷积层
    print("正在替换卷积层（GPU版本）...")
    replace_conv2d(model)
    model.eval() # 推理模式

    print("开始 GPU 推理测试...")
    start_time = time()
    correct_tmp = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # 缩放：在 CPU 上建议缩放到 32x32 或 64x64，224 太慢了
            images_resized = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
            
            output = model(images_resized)
            
            pred = output.argmax(dim=1) 
            correct = (pred == labels).sum().item() 
            correct_tmp += correct
            total += labels.size(0)
            
            print(f"Batch {batch_idx}, Accuracy: {correct_tmp / total:.4f}")
            
    end_time = time()
    print(f"测试完成。总耗时: {end_time - start_time:.2f} 秒")
    print(f"最终准确率: {correct_tmp / total:.4f}")



