import torch
from torch import nn, optim
import random
import numpy as np
import time

import os
# from pathlib import Path
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# from scipy.ndimage import zoom
from scipy.stats import rankdata, spearmanr

from utils import save_experiment, save_checkpoint
from vit import ViTForClassfication, EarlyFusion, LateFusion, get_list_shape
from torchvision import datasets, transforms
from data import ImageDepthDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from PIL import Image


def check_device_availability(device):
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"CUDA is available! Using {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Falling back to CPU.")
            return "cpu"
    else:
        print("Using CPU.")
    return device

def spearman_rank_correlation(attention_img, attention_dpt):
    """
    :param attention_img: torch.Tensor, shape (batch, head, 65, 65) 

    """
    assert attention_img.shape == attention_dpt.shape, "入力のshapeが一致しません"

    rs_batch = []

    for idx, (entry_img, entry_dpt) in enumerate(zip(attention_img, attention_dpt)):
        img_flatten = entry_img.flatten().cpu().numpy()
        dpt_flatten = entry_dpt.flatten().cpu().numpy()

        coeff, _ = spearmanr(img_flatten, dpt_flatten)
        rs_batch.append(coeff)

    return rs_batch

def precision_top_k(attention_img, attention_dpt, k=1.0):

    precisions_batch = []
    for idx, (entry_img, entry_dpt) in enumerate(zip(attention_img, attention_dpt)):

        # calculate top-k% 
        top_k = int(len(entry_img) * k)

        img_flatten = entry_img.flatten().cpu().numpy()
        dpt_flatten = entry_dpt.flatten().cpu().numpy()

        # idx sort by descending and cut off
        img_top_k_idx = np.argsort(img_flatten)[::-1][:top_k]
        dpt_top_k_idx = np.argsort(dpt_flatten)[::-1][:top_k]

        # 
        intersection = len(set(img_top_k_idx) & set(dpt_top_k_idx))
        precision = intersection / top_k

        precisions_batch.append(precision)

    return precisions_batch

def total_consistency(attention_data, k=1.0):

    rs = []
    precisions = []

    for idx, entry in enumerate(attention_data): 
        attention_img = entry["attention_img"] 
        attention_dpt = entry["attention_dpt"]

        # remove CLS
        attention_img = attention_img[:, :, 1:, 1:].cpu() ## trans to numpy
        attention_dpt = attention_dpt[:, :, 1:, 1:].cpu()

        # averaging in head
        attention_img =torch.mean(attention_img, dim=1)  ##(batch, head, H, W) -> (batch, H, W)
        attention_dpt =torch.mean(attention_dpt, dim=1)

        rs_batch = spearman_rank_correlation(attention_img, attention_dpt)
        precision_top_k_batch = precision_top_k(attention_img, attention_dpt, k)

        precisions.extend(precision_top_k_batch)
        rs.extend(rs_batch)

    return rs, precisions




config = {
    "patch_size": 32,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 24,  # changed 48->24
    "num_hidden_layers": 8,
    "num_attention_heads": 6,
    "intermediate_size": 4 * 24,  # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 256,
    "num_classes": 10,  # num_classes of CIFAR10
    "num_channels": 3,
    "num_channels_forDepth": 1,
    "qkv_bias": True,
    "use_faster_attention": True,
    "use_method1": True
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config["intermediate_size"] == 4 * config["hidden_size"]
assert config["image_size"] % config["patch_size"] == 0

label_mapping = {}

# Image only ViT
class SimpleViT_loss:
    def __init__(self, model, images, depth, labels, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.images = images
        self.depth = depth
        self.labels = labels

    def calculate_loss(self):
        return self.loss_fn(self.model(self.images)[0], self.labels)

# EarlyFusion
class Early_loss:
    def __init__(self, model, images, depth, labels, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.images = images
        self.depth = depth
        self.labels = labels

    def calculate_loss(self):
        preds = self.model(self.images, self.depth)[0]
        loss = self.loss_fn(preds, self.labels)
        # loss2 = self.loss_fn(self.model(self.depth, Isdepth=True)[0], self.labels)
        return loss
    
# LateFusion
class Late_loss:
    def __init__(self, model, images, depth, labels, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.images = images
        self.depth = depth
        self.labels = labels

    def calculate_loss(self):
        # Get predictions from the model
        preds = self.model(self.images, self.depth)[0]  # logits
        # Compute loss
        loss = self.loss_fn(preds, self.labels)

        return loss
    
def decode_label(encoded_label, label_mapping):
    return label_mapping.get(encoded_label, "Unknown")

def visualize_attention(attention_data, zoomsize=4, layer_idx=0, head_idx=0, save_path=None):
    global label_mapping
    ### ---attnmap:(batch, head, 65, 65)

    for idx, entry in enumerate(attention_data): #entry is batch image and depth pair
        if idx > 10:
            break
        image = entry["image"] 
        depth = entry["depth_image"]
        attention_img = entry["attention_img"]
        attention_dpt = entry["attention_dpt"]
        label = entry["label"]

        layer_idx = "ave"
        head_idx = "ave"
       
        # remove CLS
        attention_img = attention_img[:, :, 1:, 1:].cpu() ## trans to numpy

        # resize attentionmap
        upsample = nn.Upsample(scale_factor=(zoomsize,zoomsize), mode='nearest')  ## mode choice = {nearest, bilinear, bicubic}
        attentionMAP_img = (upsample(attention_img))

        # averaging in head
        attentionMAP_img_ave =torch.mean(attentionMAP_img, dim=1)  ##(batch, head, H, W) -> (batch, H, W)

        for i in range(len(attentionMAP_img_ave)):
            label_i = decode_label(label[i], label_mapping)
            plt.figure(figsize=(8, 6))
            plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap="gray", alpha=0.8) 
            plt.imshow(attentionMAP_img_ave[i], cmap="jet", alpha=0.5)  
            plt.title(f"RGB Attention Map for Label: {label_i}, Layer: {layer_idx}, Head: {head_idx}")
            plt.colorbar()

            if save_path:
                plt.savefig(f"{save_path}_image{idx}_layer{layer_idx}_head{head_idx}.png")
                plt.close()
            else:
                plt.show()
                plt.close()

        if attention_dpt is not None:
            # remove CLS
            attention_dpt = attention_dpt[:, :, 1:, 1:].cpu() ## trans to numpy
            # print(f"attention_dpt.shape:{attention_dpt.shape}")

            # resize attentionmap
            upsample = nn.Upsample(scale_factor=(zoomsize,zoomsize), mode='nearest')
            attentionMAP_dpt = (upsample(attention_dpt))
            # print(f"attentionMAP_dpt.shape:{attentionMAP_dpt.shape}")

            # averaging in head
            attentionMAP_dpt_ave =torch.mean(attentionMAP_dpt, dim=1)
            # print(f"attentionMAP_dpt_ave.shape:{attentionMAP_dpt_ave.shape}")

            for i in range(len(attentionMAP_dpt_ave)):
                plt.figure(figsize=(8, 6))
                plt.imshow(depth.permute(1, 2, 0).cpu().numpy(), cmap="gray", alpha=0.8) 
                plt.imshow(attentionMAP_dpt_ave[i], cmap="jet", alpha=0.5)  
                plt.title(f"Depth Attention Map for Label: {label}, Layer: {layer_idx}, Head: {head_idx}")
                plt.colorbar()

                if save_path:
                    plt.savefig(f"{save_path}_dapth{idx}_layer{layer_idx}_head{head_idx}.png")
                    plt.close()
                else:
                    plt.show()
                    plt.close()



class Trainer:
    """
    The simple trainer.
    """
    def __init__(self, model, optimizer, loss_fn, method, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.method = method
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        fusion_methods = {0: SimpleViT_loss, 1: Early_loss, 2: Late_loss}
        self.fusion_method = fusion_methods.get(method)
        self.num_layers = model.config["num_hidden_layers"]
        self.k = model.config["spearman_k"]

    def train(self, trainloader, testloader, validloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, valid_losses, accuracies, valid_accuracies = [], [], [], [], []
        # Train the model
        for i in range(epochs):
            print(f"train epoch: {i}")
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss, attention_data = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            rs, precision_top_k = total_consistency(attention_data, self.k)
        
            if validloader is not None:
                valid_accuracy, valid_loss, _ = self.evaluate(validloader)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_accuracy)
                print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"Valid loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Spearman score: {np.mean(rs):.4f}, Precision top{self.k*100}% score: {np.mean(precision_top_k):.4f}")
            else:
                print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"Spearman score: {np.mean(rs):.4f}, Precision top{self.k*100}% score: {np.mean(precision_top_k):.4f}")

            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and i + 1 != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)

        print(f"amount of pair:{len(rs)}")
        # visualize_attention
        layer_idx = 2
        head_idx = 0
        # print(f"attn img shape:{attention_img[0]}")

        ## ---- sample
        save_path = r"../ViT_scratch/sample/"
        image_size = self.model.config["image_size"]
        patch_size = self.model.config["patch_size"]
        num_patch = image_size/patch_size
        visualize_attention(attention_data, image_size/(num_patch*num_patch), layer_idx, head_idx, save_path=save_path)

        # Save the experiment
        save_experiment(
            self.exp_name, config, self.model, train_losses, test_losses, accuracies
        )

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            # [ [print(sub_t) for sub_t in t] for t in batch.values()]
            # batch = [ [sub_t.to(self.device) for sub_t in t] for t in batch.values()]
            batch = [t.to(self.device) for t in batch.values()]
            images, depth, labels = batch
            # print(
            #     f"depth shape: {depth.shape}, images shape: {images.shape}, labels shape: {labels.shape}"
            # )

            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            # NOTE: LateFusion
            if self.method == 2:
                preds = self.model(images, depth)[0]
                loss = self.loss_fn(preds, labels)               
            elif self.method == 1:
                preds = self.model(images, depth)[0]
                loss = self.loss_fn(preds, labels)
            elif self.method == 0:
                preds = self.model(images)[0]
                loss = self.loss_fn(preds, labels)
            else:
                raise ValueError(f"Unknown loss method: {self.fusion_method}")

            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        # all_attention_maps_img = []
        # all_attention_maps_dpt = [] 
        attention_data = []

        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch.values()]
                images, depth, labels = batch
                # print(f"images:{images.shape}")  ##(4, 3, 256, 256)
                # print(f"depth:{depth.shape}")  ##(4, 1, 256, 256)
                # print(f"labels:{labels}")  ##(3, 8, 6, 0) <- this is just label

                # Get predictions
                if self.method in [1,2]:
                    logits, attention_img, attention_dpt = self.model(images, depth, attentions_choice=True)

                elif self.method == 0: 
                    logits, attention_img = self.model(images, attentions_choice=True)
                    attention_dpt = None

                # print(f"logits:{logits.shape}")
                # print(f"attention_img: {len(attention_img)}")
                # print(f"attention_img: {attention_img.shape}")
                # print(f"imagesize: {images.size(0)}")
                for i in range(images.size(0)): # .size(0) is batch_size, then processing each image
                    attention_data.append({
                        "image": images[i].detach().cpu(),
                        "depth_image": depth[i].detach().cpu(),
                        "label": labels.detach().cpu(),
                        "attention_img": attention_img[i].detach().cpu(),
                        "attention_dpt": attention_dpt[i].detach().cpu()
                    })

                # Calculate the loss
                method_instance = self.fusion_method(
                    self.model, images, depth, labels, self.loss_fn
                )
                loss = method_instance.calculate_loss()
                # loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        # print(type(all_attention_maps_img))

        return accuracy, avg_loss, attention_data


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="vit-with-10-epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_data_size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save_model_every", type=int, default=0)
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="rod_sample")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay for optimizer")
    parser.add_argument("--attentionmap", type=bool, default=False, help="Visualize Attentionmap")
    parser.add_argument("--proposal1", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--topk", type=float, default=1.0)


    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dataset_path = f"../data/{args.dataset}/"
    return args


def main():
    args = parse_args()
    device = check_device_availability(args.device)
    # Training parameters
    epochs = args.epochs
    lr = args.lr
    num_labels = args.num_labels
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    method = args.method

    transform1 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

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

        return encoded_labels
    
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

        return encoded_labels

    # 画像ファイルのパスを取得 (RGBおよび深度画像)
    def load_datapath(dataset_path):
        if dataset_path=="rgbd-dataset-10k":
            image_paths = glob.glob(os.path.join(dataset_path, "train", "images", "*.png"))
            depth_paths = glob.glob(os.path.join(dataset_path, "train", "depth", "*.png"))
        else:
            image_files = glob.glob(
            os.path.join(args.dataset_path, "**", "*.png"), recursive=True
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
            random.shuffle(paired_data)  # ペアのままシャッフル
            image_paths, depth_paths = zip(*paired_data)  # シャッフル後に再分割

            # リストに戻す
            image_paths = list(image_paths)
            depth_paths = list(depth_paths)

        # ペアの整合性を確認
        assert len(image_paths) == len(depth_paths), "Image and depth paths must have the same length!"
        return image_paths, depth_paths
    
    def load_datapath_NYU(dataset_path):
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
        random.shuffle(paired_data)
        image_paths, depth_paths, labels = zip(*paired_data)

        return list(image_paths), list(depth_paths), list(labels)

    
    image_paths, depth_paths, labels =load_datapath_NYU("..\\data\\nyu_data_sample")

    # print(f"Total image files: {len(image_files)}")
    # print(args.dataset_path)
    # image_paths, depth_paths = load_datapath(args.dataset_path)

    # デバッグ用出力
    print("After shuffle:")
    print(f"First 5 image paths: {image_paths[:5]}")
    print(f"First 5 depth paths: {depth_paths[:5]}")

    # データ数制限
    image_paths = image_paths[: args.max_data_size]
    depth_paths = depth_paths[: args.max_data_size]

    print(f"Total RGB image paths: {len(image_paths)}")
    print(f"Total depth image paths: {len(depth_paths)}")

    def get_dataloader(image_paths, depth_paths, batch_size, transform, dataset_type=0, split_ratio=(0.8, 0.1, 0.1)):

        # 訓練・検証・テストデータに分割
        train_ratio, valid_ratio, test_ratio = split_ratio
        image_train, image_temp, depth_train, depth_temp = train_test_split(image_paths, depth_paths, test_size=(valid_ratio + test_ratio), random_state=42)
        image_valid, image_test, depth_valid, depth_test = train_test_split(image_temp, depth_temp, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=42)

        # ラベル取得
        if dataset_type==0:
            getlabels = getlabels_WRGBD
        elif dataset_type==1:
            getlabels = getlabels_NYU

        train_labels = getlabels(image_train)
        valid_labels = getlabels(image_valid)
        test_labels = getlabels(image_test)

        # データセット作成
        train_dataset = ImageDepthDataset(image_train, depth_train, train_labels, transform=transform)
        valid_dataset = ImageDepthDataset(image_valid, depth_valid, valid_labels, transform=transform)
        test_dataset = ImageDepthDataset(image_test, depth_test, test_labels, transform=transform)

        # DataLoader 作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        return train_loader, valid_loader, test_loader, len(set(train_labels))

    # image_train, image_test, depth_train, depth_test = train_test_split(
    #     image_paths, depth_paths, test_size=0.2, random_state=0
    # )
    dataset_type = 1
    train_loader, valid_loader, test_loader, num_labels = get_dataloader(image_paths, depth_paths, args.batch_size, transform1, dataset_type)
    # print(aa)

    # ラベル取得
    # train_labels = getlabels_WRGBD(image_train)
    # test_labels = getlabels_WRGBD(image_test)

    # train_dataset = ImageDepthDataset(
    #     image_train, depth_train, train_labels, transform=transform1
    # )
    # test_dataset = ImageDepthDataset(
    #     image_test, depth_test, test_labels, transform=transform1
    # )

    # unique_labels = np.unique(getlabels_WRGBD(image_train + image_test))
    # num_labels = len(unique_labels)

    print(f"numberof labels: {num_labels}") 

    # train_loader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    # )  # changed num_workers 2 -> 1

    # valid_loader = DataLoader(
    #     valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    # )

    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    # )

    ViT_methods = {
        0: ViTForClassfication,
        1: EarlyFusion, 
        2: LateFusion,
    }

    model_class = ViT_methods.get(method, ViTForClassfication)

    config["num_classes"] = num_labels
    config["alpha"] = args.alpha
    config["beta"] = args.beta
    config["spearman_k"] = args.topk
    model = model_class(config)

    # Create the model, optimizer, loss function and trainer
    # model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, method, args.exp_name, device=device)

    trainer.train(
        train_loader,
        test_loader,
        valid_loader,
        epochs,
        save_model_every_n_epochs=save_model_every_n_epochs,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()  # 終了時刻を記録
    execution_time = end_time - start_time  # 実行時間を計算
    print(f"Time: {execution_time:.4f} seconds")
