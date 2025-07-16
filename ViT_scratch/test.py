import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_experiment 
from data import ImageDepthDataset, load_datapath_NYU, getlabels_NYU
from torchvision import transforms
import argparse
from train import spearman_rank_correlation
from PIL import Image
import glob
import os
import numpy as np
import math
import matplotlib.pyplot as plt

def process_attention(attention_maps, img_h, img_w):
    attention_maps = attention_maps[:, :, 0, 1:].mean(dim=1)
    size = int(math.sqrt(attention_maps.size(-1)))
    attention_maps = attention_maps.view(-1, size, size)
    attention_maps = F.interpolate(attention_maps.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(1)
    return attention_maps

def show_images_with_attention(raw_images, raw_depths, attention_maps_img, attention_maps_dpt, label_ids, predictions, id_to_label, output=None, show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    num_images = len(raw_images)
    img_h, img_w = raw_images[0].shape[:2]

    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([
        np.ones((img_h, img_w)),        # 1: RGB (非表示)
        np.zeros((img_h, img_w)),       # 2: RGB (表示)
        np.ones((img_h, img_w)),        # 3: Depth (非表示)
        np.zeros((img_h, img_w)),       # 4: Depth (表示)
    ], axis=1)
    raw_depths = [np.stack([d]*3, axis=2) if d.ndim==2 else d for d in raw_depths]

    for i in range(num_images):
        ax = fig.add_subplot(int(num_images/3), 3, i+1, xticks=[], yticks=[])
        # Depth画像を3chに変換
        
        img = np.concatenate((raw_images[i], raw_images[i], raw_depths[i], raw_depths[i]), axis=1)
        ax.imshow(img)

        extended_attention_map = np.concatenate([
            np.zeros((img_h, img_w)),                           # 1: RGB (非表示)
            attention_maps_img[i].cpu().detach().numpy(),       # 2: RGB (表示)
            np.zeros((img_h, img_w)),                           # 3: Depth (非表示)
            attention_maps_dpt[i].cpu().detach().numpy(),       # 4: Depth (表示)
        ], axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')

        gt = id_to_label[str(label_ids[i])]
        pred = id_to_label[str(predictions[i].item())]
        ax.set_title(
            f"gt: {gt} / pred: {pred}", 
            fontsize=12,
            color=("green" if gt == pred else "red"))
    
    plt.tight_layout()
    if output is not None:
        plt.savefig(output)
    if show:
        plt.show()

import torch
import torch.nn as nn

class VariationalMI(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, y):
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)
        var = logvar.exp()

        log_p = -0.5 * (((y - mu)**2)/var + logvar + torch.log(torch.tensor(2*3.1415))).sum(dim=1)

        y_perm = y[torch.randperm(y.size(0))]
        log_p_neg = -0.5 * (((y_perm - mu)**2)/var + logvar + torch.log(torch.tensor(2*3.1415))).sum(dim=1)

        mi = (log_p - log_p_neg).mean()
        return mi

    
def RGB_visualize_attention_NYU(model, base_path, label_mapping, image_size, output=None, device="cuda"):
    from PIL import Image
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.jpg')))
    labels, _ = getlabels_NYU(image_paths)

    num_images = 30
    indices = torch.randperm(len(image_paths))[:num_images]
    raw_images = [np.asarray(Image.open(image_paths[i]).convert("RGB")) for i in indices]
    label_ids = [labels[i] for i in indices]
    id_to_label = label_mapping
    # print(label_mapping)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = torch.stack([test_transform(Image.fromarray(img)) for img in raw_images]).to(device)

    img_h, img_w = raw_images[0].shape[:2]

    model = model.to(device)
    logits, attention_maps = model(images, attentions_choice=True)
    predictions = torch.argmax(logits, dim=1)
    attention_maps = torch.cat(attention_maps, dim=1)
    attention_maps = attention_maps[:, :, 0, 1:].mean(dim=1)
    size = int(math.sqrt(attention_maps.size(-1)))
    attention_maps = attention_maps.view(-1, size, size)
    attention_maps = F.interpolate(attention_maps.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(1)  

    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((img_h, img_w)), np.zeros((img_h, img_w))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)

        extended_attention_map = np.concatenate((np.zeros((img_h, img_w)), attention_maps[i].cpu().detach().numpy()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')

        gt = id_to_label[str(label_ids[i])]
        pred = id_to_label[str(predictions[i].item())]
        ax.set_title(
            f"gt: {gt} / pred: {pred}", 
            fontsize=8,
            color=("green" if gt == pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()

def RGBD_visualize_attention_NYU(model, base_path, label_mapping, image_size, output=None, device="cuda"):
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.png')))
    labels, _ = getlabels_NYU(image_paths)

    num_images = 12
    indices = torch.randperm(len(image_paths))[:num_images] ## if random is not desired comment out
    
    raw_images = [np.asarray(Image.open(image_paths[i]).convert("RGB")) for i in indices]
    raw_depths = [np.asarray(Image.open(depth_paths[i])) for i in indices]
    
    label_ids = [labels[i] for i in indices]
    id_to_label = label_mapping 
    # print(label_mapping)

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    images = torch.stack([test_transform(Image.fromarray(img)) for img in raw_images]).to(device)
    depths = torch.stack([depth_transform(Image.fromarray(dpt)) for dpt in raw_depths]).to(device)

    img_h, img_w = raw_images[0].shape[:2]

    model = model.to(device)
    logits, attention_maps_img, attention_maps_dpt = model(images, depths, attentions_choice=True)
    predictions = torch.argmax(logits, dim=1)

    ## averaging in block  4*[12, 4, 65, 65] -> [12, 4, 65, 65]
    attention_maps_img = torch.stack(attention_maps_img, dim=0).mean(dim=0)
    attention_maps_dpt = torch.stack(attention_maps_dpt, dim=0).mean(dim=0)
    
    attention_maps_img = process_attention(attention_maps_img, img_h, img_w)
    attention_maps_dpt = process_attention(attention_maps_dpt, img_h, img_w)


    show_images_with_attention(
        raw_images, raw_depths, attention_maps_img, attention_maps_dpt,
        label_ids, predictions, id_to_label, "attention.png", True
    )

def test(model, base_path, label_mapping, image_size, device="cuda", num_images = 16): 
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.png')))
    labels, _ = getlabels_NYU(image_paths)

    indices = torch.randperm(len(image_paths))[:num_images] ## if random is not desired comment out
    
    raw_images = [np.asarray(Image.open(image_paths[i]).convert("RGB")) for i in indices]
    raw_depths = [np.asarray(Image.open(depth_paths[i])) for i in indices]
    
    label_ids = [labels[i] for i in indices]
    id_to_label = label_mapping 

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    images = torch.stack([test_transform(Image.fromarray(img)) for img in raw_images]).to(device)
    depths = torch.stack([depth_transform(Image.fromarray(dpt)) for dpt in raw_depths]).to(device)

    model = model.to(device)
    logits, attention_maps_img, attention_maps_dpt = model(images, depths, attentions_choice=True)
    predictions = torch.argmax(logits, dim=1)
    img_h, img_w = raw_images[0].shape[:2]

    for i, (attn_img, attn_dpt) in enumerate(zip(attention_maps_img, attention_maps_dpt)):
        # attn_img_np = attn_img.cpu().detach().numpy()  ## [batch, head, 65, 65]
        # attn_dpt_np = attn_dpt.cpu().detach().numpy()  
        attn_img = process_attention(attn_img, img_h, img_w) ## [batch, head, 65, 65] -> 
        attn_dpt = process_attention(attn_dpt, raw_depths[0].shape[0], raw_depths[0].shape[1])
        # print(f"attn_img shape: {attn_img.shape}, attn_dpt shape: {attn_dpt.shape}")
        # print(a)

        score_s = spearman_rank_correlation(attn_img, attn_dpt)
        print(f"Spearman Rank Correlation for block {i}: {sum(score_s) / len(score_s):.4f}")

        output_name = f"attention_block{i}.png"
        show_images_with_attention(
            raw_images, raw_depths, attn_img, attn_dpt,
            label_ids, predictions, id_to_label, output_name, False
        )
        


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = r'..\data\nyu_data\nyu2'
    experiment_name = "vit-with-10-epochs"
    map_location = "cuda" if torch.cuda.is_available() else "cpu"

    config, model, _, _, _, label_mapping = load_experiment(
        experiment_name,
        checkpoint_name='RGB_10_model_final.pt',
        depth=False,
        map_location=map_location
        )
    image_size = config['image_size']
    # RGB_visualize_attention_NYU(model, base_path, label_mapping, image_size, "attention.png", device=device)

    config, model, _, _, _, label_mapping = load_experiment(
        experiment_name,
        checkpoint_name='sharefusion_30epoch.pt',
        depth=True,
        map_location=map_location
        )
    image_size = config['image_size']
    # RGBD_visualize_attention_NYU(model, base_path, label_mapping, image_size, "sharefusion_attention.png", device=device)

    test(model, base_path, label_mapping, image_size, device=device, num_images=3)


if __name__ == "__main__":
    main()

