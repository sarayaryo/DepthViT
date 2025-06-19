import torch
from utils import load_experiment  # あなたが使っているload関数に置換
from data import ImageDepthDataset, load_datapath_NYU, getlabels_NYU
from torchvision import transforms
import argparse
import glob
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize_attention_NYU(model, base_path, label_mapping, output=None, device="cuda"):
    from PIL import Image
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.jpg')))
    labels, _ = getlabels_NYU(image_paths)

    num_images = 30
    indices = torch.randperm(len(image_paths))[:num_images]
    raw_images = [np.asarray(Image.open(image_paths[i]).convert("RGB")) for i in indices]
    label_ids = [labels[i] for i in indices]
    id_to_label = label_mapping

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = torch.stack([test_transform(Image.fromarray(img)) for img in raw_images]).to(device)

    img_h, img_w = raw_images[0].shape[:2]

    model = model.to(device)
    logits, attention_maps = model(images, output_attentions=True)
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

        extended_attention_map = np.concatenate((np.zeros((img_h, img_w)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')

        gt = id_to_label[label_ids[i]]
        pred = id_to_label[predictions[i].item()]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt == pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = r'..\data\nyu_data\nyu2'
    experiment_name = "vit-with-10-epochs"
    config, model, _, _, _ = load_experiment(experiment_name)

    visualize_attention_NYU(model, base_path, label_mapping, "attention.png", device=device)

if __name__ == "__main__":
    main()

