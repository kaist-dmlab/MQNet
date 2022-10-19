from torchvision import datasets
from torch.utils.data.dataset import Dataset
import numpy as np

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        self.targets = np.array(self.cifar10.targets)
        self.classes = self.cifar10.classes

    def __getitem__(self, index):
        data, _ = self.cifar10[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)