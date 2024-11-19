import torch
import time
from torch import nn, optim
import os
from pathlib import Path
import glob
from sklearn.preprocessing import LabelEncoder

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication, DepthViT
from torchvision import datasets, transforms
from data import ImageDepthDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


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


config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 24,  # changed 48->24
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
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
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config["intermediate_size"] == 4 * config["hidden_size"]
assert config["image_size"] % config["patch_size"] == 0


# Image only ViT
class SimpleViT:
    def __init__(self, model, images, depth, labels, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.images = images
        self.depth = depth
        self.labels = labels

    def calculate_loss(self):
        print(self.images.shape)
        return self.loss_fn(self.model(self.images)[0], self.labels)


# Early Fusion
class EarlyFusionDepthViT:
    def __init__(self, model, images, depth, labels, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.images = images
        self.depth = depth
        self.labels = labels

    def calculate_loss(self):
        image_depth = torch.cat((self.images, self.depth), dim=1)
        # print(image_depth[0].shape)
        loss = self.loss_fn(self.model(image_depth)[0], self.labels)
        # エラー未解決
        return loss


# Late Fusion
class LateFusionDepthViT:
    def __init__(self, model, images, depth, labels, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.images = images
        self.depth = depth
        self.labels = labels

    def calculate_loss(self):
        #print("[test] here: loss = self.loss_fn")
        preds = self.model(self.images, self.depth)[0]
        loss = self.loss_fn(preds, self.labels)
        # loss2 = self.loss_fn(self.model(self.depth, Isdepth=True)[0], self.labels)

        return loss


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, method, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        loss_methods = {0: SimpleViT, 1: EarlyFusionDepthViT, 2: LateFusionDepthViT}
        self.loss_method = loss_methods.get(method)

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            print(f"train epoch: {i}")
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(
                f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and i + 1 != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
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
            print(
                f"depth shape: {depth.shape}, images shape: {images.shape}, labels shape: {labels.shape}"
            )

            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss

            # NOTE: LateFusionDepthViT

            print("[test] here: preds = self.model(")

            preds = self.model(images, depth)[0]

            print(f"[test] preds: {preds}")

            loss = self.loss_fn(preds, labels)

            print(f"[test] loss: {loss}")

            # loss = self.loss_fn(self.model(images)[0], labels)

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
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch.values()]
                images, depth, labels = batch

                # Get predictions
                logits, _ = self.model(images, depth)

                # Calculate the loss
                method_instance = self.loss_method(
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
        return accuracy, avg_loss


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
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    method = args.method
    # # Load the CIFAR10 dataset
    # trainloader, testloader, _ = prepare_data(batch_size=batch_size)

    # Newデータセット
    # dataset_path = r'Imagedata\desk_1\rgbd-scenes\desk\desk_1'

    transform1 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # train_dataset = datasets.ImageFolder(train_dataset_dir, transform=transform1)
    # test_dataset = datasets.ImageFolder(test_dataset_dir, transform=transform1)

    # train_list = glob.glob(os.path.join(train_dataset_dir,'**','*.png'), recursive=True)
    # test_list = glob.glob(os.path.join(test_dataset_dir, '**','*.jpg'), recursive=True)

    def getlabels(image_files):
        le = LabelEncoder()
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
            print(f"File: {image_file}, ClassLabel: {classlabel}")
        encoded_labels = le.fit_transform(labels)
        print(f"encoded_labels: {encoded_labels}")
        # print(f"label amount::::{len(labels)}")
        return encoded_labels

    # 画像ファイルのパスを取得 (RGBおよび深度画像)
    image_files = glob.glob(
        os.path.join(args.dataset_path, "**", "*.png"), recursive=True
    )
    print(f"Total image files: {len(image_files)}")
    image_paths = []
    depth_paths = []

    # 画像ファイルを RGB と深度に分類
    for file_path in image_files:
        filename = os.path.basename(file_path)
        if "depth" in filename:
            depth_paths.append(file_path)
        elif "maskcrop" not in filename:
            image_paths.append(file_path)

    min_length = min(len(image_paths), len(depth_paths))
    print(f"Total RGB image paths: {len(image_paths)}")
    print(f"Total depth image paths: {len(depth_paths)}")

    # image_paths と depth_paths で、順番が正しいかどうかは保証されない
    # image_paths = image_paths[:min_length]
    # depth_paths = depth_paths[:min_length]
    # print(f"Adjusted Total RGB image paths: {len(image_paths)}")
    # print(f"Adjusted Total depth image paths: {len(depth_paths)}")

    # データ数制限
    image_paths = image_paths[: args.max_data_size]
    depth_paths = depth_paths[: args.max_data_size]

    image_train, image_test, depth_train, depth_test = train_test_split(
        image_paths, depth_paths, test_size=0.2, random_state=0
    )
    # print("[test] image_train: ", image_train)
    # print("[test] image_test: ", image_test)
    # print("[test] depth_train: ", depth_train)
    # print("[test] depth_test: ", depth_test)
    
    # ラベル取得
    train_dataset = ImageDepthDataset(
        image_train, depth_train, getlabels(image_train), transform=transform1
    )
    test_dataset = ImageDepthDataset(
        image_test, depth_test, getlabels(image_test), transform=transform1
    )


    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )  # changed num_workers 2 -> 1

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    ViT_methods = {
        0: ViTForClassfication,
        # 1: Method1,  # 未定義
        2: DepthViT,
    }

    model_class = ViT_methods.get(method, ViTForClassfication)
    model = model_class(config)

    # Create the model, optimizer, loss function and trainer
    # model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, method, args.exp_name, device=device)

    trainer.train(
        train_loader,
        test_loader,
        epochs,
        save_model_every_n_epochs=save_model_every_n_epochs,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()  # 終了時刻を記録
    execution_time = end_time - start_time  # 実行時間を計算
    print(f"Time: {execution_time:.4f} seconds")
