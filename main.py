import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import time
from torchvision.models import resnet18

from models import SimCLR

# 设置数据路径
batch_size = 128
dropout = 0.5
num_epochs = 200
lr = 0.00001
pretrained = 1
optm = "adam"
lrsch = 0

load_epoch = 1000

device = torch.device("cuda:1")

# 定义数据预处理和增强
train_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(224),  # 随机裁剪并调整到指定大小
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，增加左右方向上的泛化
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
    transforms.RandomRotation(20),  # 随机旋转±20度
    transforms.ToTensor(),  # 转换成Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])
val_transforms = transforms.Compose([
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
# full_data = ImageFolder(root=data_path, transform=None)


# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, transform):
#         self.dataset = dataset
#         self.transform = transform

#     def __getitem__(self, index):
#         image, label = self.dataset[index]
#         return self.transform(image), label

#     def __len__(self):
#         return len(self.dataset)
cifar100_training = CIFAR100(root='../datasets/cifar100', train=True, download=True, transform=train_transforms)
cifar100_testing = CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=val_transforms)
print(len(cifar100_training), len(cifar100_testing))

# 定义 DataLoader
train_loader = DataLoader(cifar100_training, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(cifar100_testing, batch_size=batch_size, shuffle=False, num_workers=8)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


# 加载预训练的 ResNet-18 模型
if pretrained == 1:
    model = models.resnet18(weights="IMAGENET1K_V1", )
elif pretrained == 0:
    model = models.resnet18(weights=None)
elif pretrained == 2:
    base_encoder = eval('resnet18')
    simclr_model = SimCLR(base_encoder, projection_dim=256)
    simclr_model.load_state_dict(torch.load('./logs/SimCLR/cifar100/simclr_{}_epoch{}.pt'.format('resnet18', load_epoch)))
    model = simclr_model.enc
# 修改输出层大小为100
num_ftrs =models.resnet18(weights=None).fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 100),
    nn.Dropout(dropout),
)

for param in model.parameters():
    param.requires_grad = True
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
if optm == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optm == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
print(model)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
# 创建TensorBoard SummaryWriter
suffix_num = timestamp_str = str(int(time.time()))[-6:]
writer = SummaryWriter(
    log_dir=f'runs/resnet18_pretrained{pretrained}_lr{lr}_epoch{num_epochs}_dropout{dropout}_{optm}_lrsch{lrsch}_{suffix_num}')

model = model.to(device)

# 训练模型
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    train_loss = train_loss / len(train_loader.batch_sampler)
    train_accuracy = train_correct / train_total
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    # 验证模型
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)  # 不一定是数量完整的batch
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(cifar100_testing)
    test_accuracy = test_correct / test_total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.6f}, Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.6f}")
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    if lrsch:
        scheduler.step()

    # 保存最好的模型
    os.makedirs('saved_models', exist_ok=True)
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        if (pretrained and best_acc > 0.75) or (not pretrained and best_acc > 0.6):
            torch.save(model.state_dict(),
                       f'saved_models/resnet18_pretrained{pretrained}_lr{lr}_epoch{num_epochs}_dropout{dropout}_{optm}_acc{best_acc:.6f}.pth')

writer.close()