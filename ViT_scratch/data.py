# Import libraries
import torch
import os
import glob
import random
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
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
        # print(f"self.labels:{labels}")
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
            'label': label,
            'path': self.image_paths[idx],
        }


def getlabels_WRGBD(image_files):
    le = LabelEncoder()
    global label_mapping
    labels = []
    for image_file in image_files:
        # ファイル名の拡張子を除く部分を取得 (例: 'apple_1_1_1_crop')
        filename = os.path.splitext(os.path.basename(image_file))[0]
        
        # クラスラベルを抽出 ('apple_1' の部分)
        classlabel = filename.split("_")[
            0
        ]  # 'apple_1' の 'apple' 部分だけを取り出す
        # classlabel = '_'.join(filename.split('_')[:2])
        labels.append(classlabel)
        # 取得したクラスラベルとファイル名の確認 (必要に応じて処理を追加)
        # print(f"File: {image_file}, ClassLabel: {classlabel}")
    encoded_labels = le.fit_transform(labels)
    label_mapping = {index: label for index, label in enumerate(le.classes_)}

    return encoded_labels, label_mapping

def getlabels_NYU(folder_paths):
    le = LabelEncoder()
    labels = []  # 空のリストを作成
    for folder in folder_paths:
        # print(f"folder:{folder}")
        # フォルダのパスから最後の部分（フォルダ名）を取得
        dir_path = os.path.dirname(folder)
        folder_name = os.path.basename(dir_path)
        # print(f"last:{folder_name}")
        # フォルダ名を"_"で区切り、最初の要素をラベルとして取り出す
        label = folder_name.split("_")[0]
        # print(f"label:{label}")
        
        # ラベルをリストに追加
        labels.append(label)

    # ラベルを数値にエンコード
    encoded_labels = le.fit_transform(labels)

    # インデックスとラベルのマッピングを作成（必要であれば返すなどして使える）
    label_mapping = {index: label for index, label in enumerate(le.classes_)}

    return encoded_labels, label_mapping

def getlabels_TinyImageNet(folder_paths):
    le = LabelEncoder()
    labels = []  # 空のリストを作成
    for folder in folder_paths:
        # print(f"folder:{folder}")
        # print(aaa)
        # フォルダのパスから最後の部分（フォルダ名）を取得
        dir_path = os.path.dirname(folder)
        folder_name = os.path.basename(dir_path)
        # print(f"last:{folder_name}")
        # フォルダ名を"_"で区切り、最初の要素をラベルとして取り出す
        label = folder_name.split("_")[0]
        # print(f"label:{label}")
        
        # ラベルをリストに追加
        labels.append(label)

    # ラベルを数値にエンコード
    encoded_labels = le.fit_transform(labels)

    # インデックスとラベルのマッピングを作成（必要であれば返すなどして使える）
    label_mapping = {index: label for index, label in enumerate(le.classes_)}

    return encoded_labels, label_mapping



# 画像ファイルのパスを取得 (RGBおよび深度画像)
def load_datapath_WRGBD(dataset_path, random_seed=42):
    if dataset_path=="rgbd-dataset-10k":
        image_paths = glob.glob(os.path.join(dataset_path, "train", "images", "*.png"))
        depth_paths = glob.glob(os.path.join(dataset_path, "train", "depth", "*.png"))
    else:
        image_files = glob.glob(
        os.path.join(dataset_path, "**", "*.png"), recursive=True
        )
        # 画像ファイルを RGB と深度に分類
        image_paths = []
        depth_paths = []
        for file_path in image_files:
            filename = os.path.basename(file_path)
            if "depth" in filename:
                depth_paths.append(file_path)
            elif "maskcrop" not in filename:
                image_paths.append(file_path)
            # ペア化されたデータをシャッフル
        paired_data = list(zip(image_paths, depth_paths))
        random.seed(random_seed)
        random.shuffle(paired_data)  # ペアのままシャッフル
        image_paths, depth_paths = zip(*paired_data)  # シャッフル後に再分割

        # リストに戻す
        image_paths = list(image_paths)
        depth_paths = list(depth_paths)

    # ペアの整合性を確認
    assert len(image_paths) == len(depth_paths), "Image and depth paths must have the same length!"
    return image_paths, depth_paths, None


def load_datapath_NYU(dataset_path, random_seed=42):
    image_paths = []
    depth_paths = []
    labels = []

    # 各クラスのフォルダを取得
    class_folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    for folder in class_folders:
        # print(folder)
        class_label = os.path.basename(folder).split("_")[0]  # 'classroom' や 'basement' を取得
        rgb_files = sorted(glob.glob(os.path.join(folder, "*.jpg")))  # RGB画像
        depth_files = sorted(glob.glob(os.path.join(folder, "*.png")))  # 深度画像

        # RGB画像と深度画像のペアを作成
        for rgb, depth in zip(rgb_files, depth_files):
            image_paths.append(rgb)
            depth_paths.append(depth)
            labels.append(class_label)

    # シャッフル
    paired_data = list(zip(image_paths, depth_paths, labels))
    random.seed(random_seed)
    random.shuffle(paired_data)
    image_paths, depth_paths, labels = zip(*paired_data)

    return list(image_paths), list(depth_paths), list(labels)

def load_datapath_TinyImageNet(dataset_path):
    image_paths = []
    depth_paths = []
    labels = []

    # 各クラスのフォルダを取得
    class_folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    for folder in class_folders:
        # print(folder)
        
        class_label = os.path.basename(folder).split("_")[0]  # 'classroom' や 'basement' を取得
        # print(f"label:{class_label}")

        rgb_files = sorted(glob.glob(os.path.join(folder, "*_rgb.png")))  # RGB画像
        # print(f"rgb_files:{rgb_files}")
        depth_files = sorted(glob.glob(os.path.join(folder, "*_depth.png")))  # 深度画像
        # print(f"depth_files:{depth_files}")

        # RGB画像と深度画像のペアを作成
        for rgb, depth in zip(rgb_files, depth_files):
            image_paths.append(rgb)
            depth_paths.append(depth)
            labels.append(class_label)
        # print(aaa)

    # シャッフル
    paired_data = list(zip(image_paths, depth_paths, labels))
    random.seed(42)
    random.shuffle(paired_data)
    image_paths, depth_paths, labels = zip(*paired_data)

    return list(image_paths), list(depth_paths), list(labels)

def get_dataloader(image_paths, depth_paths, batch_size, transform, dataset_type=0, split_ratio=(0.8, 0.1, 0.1)):
    from sklearn.model_selection import train_test_split

    # 訓練・検証・テストデータに分割
    train_ratio, valid_ratio, test_ratio = split_ratio
    image_train, image_temp, depth_train, depth_temp = train_test_split(image_paths, depth_paths, test_size=(valid_ratio + test_ratio), random_state=42)
    image_valid, image_test, depth_valid, depth_test = train_test_split(image_temp, depth_temp, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=42)

    # ラベル取得
    if dataset_type==0:
        getlabels = getlabels_WRGBD
    elif dataset_type==1:
        getlabels = getlabels_NYU
    elif dataset_type==2:
        getlabels = getlabels_TinyImageNet

    train_labels, label_mapping_train = getlabels(image_train)
    valid_labels, _ = getlabels(image_valid)
    test_labels, label_mapping_test = getlabels(image_test)

    # データセット作成
    train_dataset = ImageDepthDataset(image_train, depth_train, train_labels, transform=transform)
    valid_dataset = ImageDepthDataset(image_valid, depth_valid, valid_labels, transform=transform)
    test_dataset = ImageDepthDataset(image_test, depth_test, test_labels, transform=transform)

    # DataLoader 作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, len(set(train_labels)), label_mapping_test