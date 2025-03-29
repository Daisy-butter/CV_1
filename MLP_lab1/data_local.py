import torchvision
import torchvision.transforms as transforms

# 自动下载 CIFAR-10 数据集到指定目录
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root=r"C:\Users\31521\OneDrive\桌面\files\academic\FDU\25春大三下\计算机视觉\lab_data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=r"C:\Users\31521\OneDrive\桌面\files\academic\FDU\25春大三下\计算机视觉\lab_data", train=False, transform=transform, download=True)
