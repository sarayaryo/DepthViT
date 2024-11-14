# Import libraries
import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

class ImageDepthDataset(Dataset):
    def __init__(self, image_paths, depth_paths, labels, transform=None):
        """
        Args:
            image_paths (list of str): 画像ファイルへのパスリスト。
            depth_paths (list of str): 深度画像ファイルへのパスリスト。
            labels (list of int): 各画像のラベル。
            transform (callable, optional): 画像に適用する変換。
        """
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)
    
    def extract_class_label(self, path):
        # ファイル名から「desk_1」のような基本カテゴリを取得
        basename = os.path.basename(path)
        label = basename.split('_')[0] + "_" + basename.split('_')[1] 
        return label
    
    def __getitem__(self, idx):
        # 画像をロード
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # 深度情報をロード
        depth = Image.open(self.depth_paths[idx]).convert("L")  # 深度はグレースケール（L）で読み込み

        label = self.labels[idx]


        # 変換を適用
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        # データを辞書形式で返す
        return {
            'image': image,
            'depth': depth,
            'label': label
        }