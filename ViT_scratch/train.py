import torch
from torch import nn, optim
import numpy as np
import gc
import sys
import csv
import psutil
import logging
import time
import os

# from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr

from utils import save_experiment, save_checkpoint
from vit import ViTForClassfication, EarlyFusion, LateFusion, get_list_shape
from torchvision import datasets, transforms
from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader

from torch.utils.data import DataLoader
import cv2 
# from PIL import Image

def normalize_attention_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    if max_val - min_val > 0:
        return (attention_map - min_val) / (max_val - min_val)
    else:
        return attention_map

def check_gpu_memory(string="", epoch=None):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB単位
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB単位
        log_message = f"{string}:  epoch(batch){epoch}:  GPUメモリ使用量: {allocated:.2f} MB / 確保済み: {reserved:.2f} MB"
        logging.info(log_message)
        # print(log_message)  # 必要ならターミナルにも出力
    else:
        logging.info("CUDAが利用できません。")
        print("CUDAが利用できません。")

def check_cpu_memory(string="", epoch=None):
    memory_info = psutil.virtual_memory()
    log_message = (
        f"{string}:  epoch(batch){epoch}:  CPUメモリ使用量: {memory_info.percent:.2f}%, "
        f"全メモリ: {memory_info.total / (1024 ** 3):.2f} GB, "
        f"使用中メモリ: {memory_info.used / (1024 ** 3):.2f} GB, "
        f"空きメモリ: {memory_info.available / (1024 ** 3):.2f} GB"
    )
    logging.info(log_message)
    # print(log_message)  # 必要ならターミナルにも出力


def save_attention_data(attention_data, save_path, filename_prefix, layer_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, data in enumerate(attention_data):
        filename = os.path.join(save_path, f"{filename_prefix}_{idx}_{layer_name}.pt")
        # 辞書形式でテンソルを保存
        torch.save(data, filename)
        # print(f"Saved: {filename}")

def save_sample_list(sample_list, filename):
    if not sample_list:
        print(f"No data to save for {filename}")
        return
    keys = sample_list[0].keys()
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sample_list)
    print(f"Saved {len(sample_list)} samples to {filename}")

def check_device_availability(device):
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"CUDA is available! Using {torch.cuda.get_device_name(0)}")

        else:
            print("CUDAが利用できません。2")
            print("CUDA is not available. Falling back to CPU.")
            return "cpu"
    else:
        print("CUDAが利用できません。")
        print("Using CPU.")
    return device


def spearman_rank_correlation(attention_img, attention_dpt):
    """
    Computes Spearman's rank correlation coefficient batch-wise.
    
    :param attention_img: torch.Tensor or np.ndarray, shape (batch, head, 65, 65)
    :param attention_dpt: torch.Tensor or np.ndarray, shape (batch, head, 65, 65)
    :return: list of Spearman's rho per entry
    """
    assert attention_img.shape == attention_dpt.shape, "入力のshapeが一致しません"

    rs_batch = []

    is_tensor = torch.is_tensor(attention_img)

    for idx, (entry_img, entry_dpt) in enumerate(zip(attention_img, attention_dpt)):
        if is_tensor:
            # detach before converting to numpy
            img_flatten = entry_img.reshape(-1).detach().cpu().numpy()
            dpt_flatten = entry_dpt.reshape(-1).detach().cpu().numpy()
        else:
            img_flatten = entry_img.reshape(-1)
            dpt_flatten = entry_dpt.reshape(-1)

        coeff, _ = spearmanr(img_flatten, dpt_flatten)
        rs_batch.append(coeff)

    return rs_batch

def precision_top_k(attention_img, attention_dpt, k=1.0):

    precisions_batch = []
    for idx, (entry_img, entry_dpt) in enumerate(zip(attention_img, attention_dpt)):

        # calculate top-k% 
        top_k = int(len(entry_img) * k)

        img_flatten = entry_img.flatten()
        dpt_flatten = entry_dpt.flatten()

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
        # print(attention_img.shape) --> torch.Size([6, 65, 65])
        attention_img = attention_img[:, 1:, 1:] ## trans to numpy
        attention_dpt = attention_dpt[:, 1:, 1:]

        # averaging in head
        attention_img =np.mean(attention_img, axis=0)  ##(head, H, W) -> (H, W)
        attention_dpt =np.mean(attention_dpt, axis=0)

        rs_batch = spearman_rank_correlation(attention_img, attention_dpt)
        precision_top_k_batch = precision_top_k(attention_img, attention_dpt, k)

        precisions.extend(precision_top_k_batch)
        rs.extend(rs_batch)

    return rs, precisions

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48, 
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,  # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,  # num_classes of CIFAR10
    "num_channels": 3,
    "num_channels_forDepth": 1,
    "qkv_bias": True,
    "use_faster_attention": True,
    "use_method1": True,  ## share-fusion(late-fusion)
    "use_method3": False, ## agreement-refined
    "learnable_alpha_beta": True,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config["intermediate_size"] == 4 * config["hidden_size"]
assert config["image_size"] % config["patch_size"] == 0

label_mapping = {}

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

    
def decode_label(encoded_label, label_mapping):
    if isinstance(encoded_label, np.ndarray):
        encoded_label = encoded_label.item()
    return label_mapping.get(encoded_label, "Unknown")


def visualize_attention(attention_data, zoomsize=4, layer_idx=0, head_idx=0, save_path=None):
    global label_mapping
    ## ---attnmap:(batch_size, head, 65, 65)
    # print(f"save_path = {repr(save_path)}")

    for idx, entry in enumerate(attention_data): #entry is batch image and depth pair
        if idx > 30:
            break
        image = entry["image"] 
        depth = entry["depth_image"]
        attention_img = entry["attention_img"]
        attention_dpt = entry["attention_dpt"]
        label = entry["label"]
        paths = entry["path"] if "path" in entry else None
        # print(f"label:{label.shape}")

        layer_idx = "ave"
        head_idx = "ave"
       
        ## attention from CLSToken
        ## attnmap:(head, 65, 65) -> (head, 64)        
        attention_img = attention_img[:, 0, 1:]
        # print(f"attention_img.shape:{attention_img.shape}")        

        ## averaging in head
        ##(head, 64) -> (64)
        attention_img_ave =np.mean(attention_img, axis=0)  
        # print(f"attentionMAP_img_ave.shape:{attention_img_ave.shape}")
        patch_count = attention_img_ave.shape[0]
        patch = int(patch_count**0.5)

        ##(64) -> (8, 8)
        attention_img_ave = normalize_attention_map(attention_img_ave)
        attention_img_ave = attention_img_ave.reshape(patch, patch)
        # print(f"attention_img_ave.shape:{attention_img_ave.shape}")

        ## resize attentionmap (8, 8) -> (256, 256)
        attention_img_ave = cv2.resize(attention_img_ave, (256, 256), interpolation=cv2.INTER_LINEAR)

        ## label_mapping
        label_i = decode_label(label, label_mapping)

        plt.figure(figsize=(8, 6))
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray) and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        plt.imshow(image, alpha=1.0) 
        plt.imshow(attention_img_ave, cmap="jet", alpha=0.5)  
        plt.title(f"RGB Attention Map for Label: {label_i}, Layer: {layer_idx}, Head: {head_idx}")
        plt.colorbar()

        if save_path:
            filename = f"{save_path}image{idx}_layer{layer_idx}_head{head_idx}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            plt.close()
        
        if attention_dpt is not None:
            ## attention from CLSToken
            ## attnmap:(head, 65, 65) -> (head, 64)        
            attention_dpt = attention_dpt[:, 0, 1:]
            # print(f"attention_dpt.shape:{attention_dpt.shape}")        

            ## averaging in head
            ##(head, 64) -> (64)
            attention_dpt_ave =np.mean(attention_dpt, axis=0)  
            # print(f"attentionMAP_dpt_ave.shape:{attention_dpt_ave.shape}")
            patch_count = attention_dpt_ave.shape[0]
            patch = int(patch_count**0.5)

            ##(64) -> (8, 8)
            attention_dpt_ave = normalize_attention_map(attention_dpt_ave)
            attention_dpt_ave = attention_dpt_ave.reshape(patch, patch)
            # print(f"attention_dpt_ave.shape:{attention_dpt_ave.shape}")

            ## resize attentionmap (8, 8) -> (256, 256)
            attention_dpt_ave = cv2.resize(attention_dpt_ave, (256, 256), interpolation=cv2.INTER_LINEAR)

            ## label_mapping
            label_i = decode_label(label, label_mapping)

            plt.figure(figsize=(8, 6))
            if isinstance(depth, torch.Tensor):
                depth = depth.permute(1, 2, 0).cpu().numpy()
            elif isinstance(depth, np.ndarray) and depth.shape[0] in [1, 3]:
                depth = depth.transpose(1, 2, 0)
            plt.imshow(depth, alpha=1.0) 
            plt.imshow(attention_dpt_ave, cmap="jet", alpha=0.5)  
            plt.title(f"Depth Attention Map for Label: {label_i}, Layer: {layer_idx}, Head: {head_idx}")
            plt.colorbar()
            if save_path:
                filename = f"{save_path}depth{idx}_layer{layer_idx}_head{head_idx}.png"
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()
                plt.close()


def process_attention_data(images, depth, labels, attention_img, attention_dpt, layer_size, image_paths=None):
    """
    Process attention data for initial, mid, and final layers.
    """
    attention_data = []
    for i in range(images.size(0)):
        attention_data.append({
            "image": images[i].detach().cpu().numpy(),
            "depth_image": depth[i].detach().cpu().numpy(),
            "label": labels[i].item(),
            "attention_img": attention_img[layer_size][i].detach().cpu().numpy(),
            "attention_dpt": attention_dpt[layer_size][i].detach().cpu().numpy() if attention_dpt is not None else None,
            "path": image_paths[i] if image_paths is not None else None
        })
    return attention_data


class Trainer:
    """
    The simple trainer.
    """
    def __init__(self, model, optimizer, loss_fn, method, exp_name, device, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.method = method
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        fusion_methods = {0: SimpleViT_loss, 1: Early_loss, 2: Late_loss}
        self.fusion_method = fusion_methods.get(method)
        self.num_layers = model.config["num_hidden_layers"]
        self.k = model.config["precision_k"]
        self.scheduler = scheduler

    def train(self, trainloader, testloader, validloader, epochs, patience, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, valid_losses, accuracies, valid_accuracies = [], [], [], [], []

        best_valid_loss = float("inf")
        # Train the model
        torch.cuda.empty_cache()
        for i in range(epochs):
            # print(f"train epoch: {i}")
            train_loss = self.train_epoch(trainloader)
            print(f"train epoch: {i}, Train loss: {train_loss:.4f}")
            logging.info(f"train epoch: {i}, Train loss: {train_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.5f}")


            if i%5 == 0:
                logging.info(f"epoch:{i}")
                check_cpu_memory("before_valid",i)
                check_gpu_memory("before_valid",i)

                ## --------------valid phase-----------------
                if validloader is not None:
                    valid_accuracy, valid_loss, _, _, _ = self.evaluate(validloader, False)
                    valid_losses.append(valid_loss)
                    valid_accuracies.append(valid_accuracy)
                    print(f"Valid loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
                    logging.info(f"Valid loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
                    
                    # Early stopping logic
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        patience_counter = 0  # Reset patience counter
                        print(f"Validation loss improved to {valid_loss:.4f}.")
                        logging.info(f"Validation loss improved to {valid_loss:.4f}.")
                        # Save the best model
                        save_checkpoint(self.exp_name, self.model, i + 1)
                    else:
                        patience_counter += 1
                        print(f"No improvement in validation loss for {patience_counter} epochs.")
                        logging.info(f"No improvement in validation loss for {patience_counter} epochs.")

                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        logging.info("Early stopping triggered.")
                        break

                else:
                    continue

                check_cpu_memory("after_valid",i)
                check_gpu_memory("after_valid",i)

            train_losses.append(train_loss)
            if getattr(self, "scheduler", None) is not None:
                self.scheduler.step()
        
            # print(f"attention_data_final.shape:{attention_data_final[1]}")
            
            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and i + 1 != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        
        ## --------------test phase-----------------
        # accuracy, test_loss, attention_data_initial, attention_data_mid, attention_data_final = self.evaluate(testloader, True)
        accuracy, test_loss, attention_data_final, wrong_images, correct_images = self.evaluate(testloader, True)
        if self.method in [1,2]:
            rs, precision_top_k = total_consistency(attention_data_final, self.k)

        test_losses.append(test_loss)
        accuracies.append(accuracy)

        save_sample_list(wrong_images, "wrong_images.csv")
        save_sample_list(correct_images, "correct_images.csv")

        if self.method in [1,2]:
            print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Spearman score: {np.mean(rs):.4f}, Precision top Attn80% score: {np.mean(precision_top_k):.4f}")
            logging.info(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Spearman score: {np.mean(rs):.4f}, Precision top Attn80% score: {np.mean(precision_top_k):.4f}")
            print(f"amount of pair:{len(rs)}")
            logging.info(f"amount of pair:{len(rs)}")
        else:
            print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            logging.info(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # visualize_attention
        layer_idx = 2
        head_idx = 0
        # print(f"attn img shape:{attention_img[0]}")

        ## ------------ save attention data --------------
        save_path_attention = "../ViT_scratch/experiments/attention/"
        # save_attention_data(attention_data_initial, save_path_attention, "attention_initial", "initial")
        # save_attention_data(attention_data_mid, save_path_attention, "attention_mid", "mid")
        # save_attention_data(attention_data_final, save_path_attention, "attention_final", "final")

        ## ---- sample
        save_path = "../ViT_scratch/sample/"
        image_size = self.model.config["image_size"]
        patch_size = self.model.config["patch_size"]
        num_patch = image_size/patch_size
        visualize_attention(attention_data_final, image_size/(num_patch*num_patch), layer_idx, head_idx, save_path=save_path)

        # Save the experiment
        save_experiment(
            self.exp_name, config, self.model, train_losses, test_losses, accuracies, label_mapping
        )

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        if self.model.config.get("learnable_alpha_beta", False):
            for i, block in enumerate(self.model.encoder_rgb_depth.blocks):
                    alpha, beta = block.attention.get_alpha_beta()
                    logging.info(f"Block {i}: alpha={alpha.item():.4f}, beta={beta.item():.4f}")
        
        for idx, batch in enumerate(trainloader):

            # batch = [ [sub_t.to(self.device) for sub_t in t] for t in batch.values()]
            # batch = [t.to(self.device) for t in batch.values()]
            # images, depth, labels = batch
            images = batch["image"].to(self.device)
            depth = batch["depth"].to(self.device)
            labels = batch["label"].to(self.device)

            # check_gpu_memory(f"batch{idx}",idx)

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

            del batch, images, depth, labels, preds, loss
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader, attentions_choice=False, infer_mode=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        attention_data_final = []
        CLS_tokens = []

        correct_images = []
        wrong_images = []

        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                images = batch["image"].to(self.device)
                depth = batch["depth"].to(self.device)
                labels = batch["label"].to(self.device)
                paths = batch["path"]

                # print(f"images:{images.shape}")  ##(batchsize, 3, 256, 256)
                # print(f"depth:{depth.shape}")  ##(batchsize, 1, 256, 256)
                # print(f"labels:{labels}")  ##(3, 8, 6, 0) <- this is just label
                if idx%100 == 0:
                    logging.info(f"Batch:{idx}")
                    check_cpu_memory("test",idx)
                    check_gpu_memory("test",idx)
                    
                # Get predictions
                if self.method in [1,2]:
                    if infer_mode:
                        logits, output_img, output_dpt, attention_img, attention_dpt = self.model(images, depth, attentions_choice=True)

                        device = self.device
                          
                        ## print(f"output_img.shape:{output_img.shape}, output_dpt.shape:{output_dpt.shape}") ([16, 65, 48])

                        f_r = output_img[:, 0, :].to(device)
                        f_d = output_dpt[:, 0, :].to(device)

                        CLS_tokens.append((f_r, f_d))

                    else:
                        logits, _, _, attention_img, attention_dpt = self.model(images, depth, attentions_choice=True)

                elif self.method == 0: 
                    logits, attention_img = self.model(images, attentions_choice=True)
                    attention_dpt = None
                
                if attentions_choice:
                    layer_size = len(attention_img)

                    # final part of layer
                    attention_data_final.extend(process_attention_data(images, depth, labels, attention_img, attention_dpt, layer_size-1, image_paths=paths))

                # Calculate the loss
                method_instance = self.fusion_method(
                    self.model, images, depth, labels, self.loss_fn
                )
                loss = method_instance.calculate_loss()
                # loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                
                ## record correct and wrong samples
                for i in range(len(labels)):
                    sample_info = {
                        "index": idx * testloader.batch_size + i,
                        "path": batch["path"][i] if "path" in batch else None,
                        "label": labels[i].item(),
                        "pred": predictions[i].item(),
                        
                    }
                    if predictions[i] == labels[i]:
                        correct_images.append(sample_info)
                    else:
                        wrong_images.append(sample_info)


                correct += torch.sum(predictions == labels).item()

                del batch, images, depth, labels, logits, attention_img, attention_dpt, loss
                gc.collect()
                torch.cuda.empty_cache()

        
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        # print(type(all_attention_maps_img))
    
        if infer_mode:
            return accuracy, CLS_tokens, avg_loss, attention_data_final, wrong_images, correct_images

        if attentions_choice:
            # return accuracy, avg_loss, attention_data_initial, attention_data_mid, attention_data_final
            return accuracy, avg_loss, attention_data_final, wrong_images, correct_images
        else:
            return accuracy, avg_loss, None, None, None


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="vit-with-10-epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_data_size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_model_every", type=int, default=10)
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="rod_sample")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay for optimizer")
    parser.add_argument("--attentionmap", type=bool, default=False, help="Visualize Attentionmap")
    parser.add_argument("--proposal1", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--topk", type=float, default=0.1)
    parser.add_argument("--dataset_type", type=int, default=1)

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dataset_path = f"../data/{args.dataset}/"
    return args


def main():
    global label_mapping
    args = parse_args()

    args.device = check_device_availability(args.device)
    with open("training.log", "w", encoding="utf-8") as f:
        f.write("")  # ファイルを空にする
    # Training parameters
    epochs = args.epochs
    lr = args.lr
    num_labels = args.num_labels
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    method = args.method

    transform1 = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    
    dataset_type = args.dataset_type
    # image_paths, depth_paths, labels =load_datapath_NYU("..\\data\\nyu_data_sample")
    if dataset_type==0:
        load_datapath = load_datapath_WRGBD
    elif dataset_type==1:
        load_datapath = load_datapath_NYU
    elif dataset_type==2:
        load_datapath = load_datapath_TinyImageNet

    image_paths, depth_paths, labels =load_datapath(args.dataset_path)

    # デバッグ用出力
    print("After shuffle:")
    print(f"First 5 image paths: {image_paths[:5]}")
    print(f"First 5 depth paths: {depth_paths[:5]}")

    # データ数制限
    image_paths = image_paths[: args.max_data_size]
    depth_paths = depth_paths[: args.max_data_size]

    print(f"Total RGB image paths: {len(image_paths)}")
    print(f"Total depth image paths: {len(depth_paths)}")

    dataset_type = args.dataset_type
    train_loader, valid_loader, test_loader, num_labels, label_mapping = get_dataloader(image_paths, depth_paths, args.batch_size, transform1, dataset_type)

    print(f"numberof labels: {num_labels}") 

    ViT_methods = {
        0: ViTForClassfication,
        1: EarlyFusion, 
        2: LateFusion,
    }

    model_class = ViT_methods.get(method, ViTForClassfication)

    config["num_classes"] = num_labels
    config["alpha"] = args.alpha
    config["beta"] = args.beta
    config["precision_k"] = args.topk
    model = model_class(config)

    # print("After model init:")
    # print(torch.cuda.memory_summary())

    # Create the model, optimizer, loss function and trainer
    # model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    trainer = Trainer(model, optimizer, loss_fn, method, args.exp_name, device=device, scheduler=scheduler)
    # trainer = Trainer(model, optimizer, loss_fn, method, args.exp_name, device=device)
    
    
    log_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    trainer.train(
        train_loader,
        test_loader,
        valid_loader,
        epochs,
        patience=3,
        save_model_every_n_epochs=save_model_every_n_epochs,
    )
    


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()  # 終了時刻を記録
    execution_time = end_time - start_time  # 実行時間を計算
    print(f"Time: {execution_time:.4f} seconds")
    logging.info(f"Time: {execution_time:.4f} seconds")
