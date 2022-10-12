from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch import tensor, long

class MyImageNet(Dataset):
    def __init__(self, file_path, transform=None, resolution=224):
        self.transform = transform
        self.resolution = resolution
        if self.resolution == 224: #32, 64, 128
            self.data = ImageFolder(file_path)
        else:
            print("Resizing Initial Data into {}x{}".format(self.resolution, self.resolution))
            transform_resize = T.Resize(size=(self.resolution,self.resolution)) #reduce the resolution once at an initial point
            self.data = ImageFolder(file_path, transform_resize)

        self.classes = self.data.classes
        self.targets = self.data.targets

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img, label = self.data[index]

        # TODO:
        #target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label#, index

    def __len__(self):
        return len(self.data)


def get_augmentations_32(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    return train_transform, test_transform

def get_augmentations_224(T_normalize):
    train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T_normalize])

    return train_transform, test_transform

def ImageNet(args):
    channel = 3
    im_size = (224, 224)
    num_classes = 1000
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    T_normalize = T.Normalize(mean, std)

    if args.resolution == 32:
        train_transform, test_transform = get_augmentations_32(T_normalize)
    if args.resolution == 224:
        train_transform, test_transform = get_augmentations_224(T_normalize)

    dst_train = MyImageNet(args.data_path+'/imagenet/train/', transform=train_transform, resolution=args.resolution)
    dst_test = MyImageNet(args.data_path+'/imagenet/val/', transform=test_transform, resolution=args.resolution)


    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test