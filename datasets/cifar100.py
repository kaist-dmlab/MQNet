from torchvision import datasets
from torch.utils.data.dataset import Dataset
import numpy as np

class MyCIFAR100(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar100 = datasets.CIFAR100(file_path, train=train, download=download, transform=transform)
        self.targets = np.array(self.cifar100.targets)
        self.classes = self.cifar100.classes

    def __getitem__(self, index):
        data, _ = self.cifar100[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)