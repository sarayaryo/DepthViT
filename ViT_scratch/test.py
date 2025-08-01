import torch
import torch.nn.functional as F
from module import VariationalMI
from utils import load_experiment
from data import ImageDepthDataset, load_datapath_NYU, getlabels_NYU
from torchvision import transforms

from train import spearman_rank_correlation, total_consistency, process_attention_data
from PIL import Image
import glob
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def process_attention(attention_maps, img_h, img_w):
    attention_maps = attention_maps[:, :, 0, 1:].mean(dim=1)
    size = int(math.sqrt(attention_maps.size(-1)))
    attention_maps = attention_maps.view(-1, size, size)
    attention_maps = F.interpolate(attention_maps.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(1)
    return attention_maps

def show_images_with_attention(raw_images, raw_depths, attention_maps_img, attention_maps_dpt, label_ids, predictions, id_to_label, output=None, show=True, i=0):

    num_images = len(raw_images)
    img_h, img_w = raw_images[0].shape[:2]

    ## figsize (640* 4, 480) == (16, 3)
    fig = plt.figure(figsize=(40, 30))
    
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
        if True:
            ax.set_title(
                f"gt: {gt} / pred: {pred}", 
                fontsize=45,
                pad = 4,
                color=("green" if gt == pred else "red")
                )
    
    # plt.tight_layout()
    if output is not None:
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0, wspace=0)
        plt.savefig(output)
    if show:
        plt.show()


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

def RGBD_visualize_attention_NYU(model, test_image_path, test_depth_path, label_mapping, image_size, output=None, device="cuda"):
    model.eval()

    # image_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.jpg')))
    # depth_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.png')))
    image_paths = test_image_path
    depth_paths = test_depth_path
    labels, _ = getlabels_NYU(image_paths)

    num_images = 12
    indices = list(range(num_images))
    # indices = torch.randperm(len(image_paths))[:num_images] ## if random is not desired comment out
    
    
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
    logits,  _, _, attention_maps_img, attention_maps_dpt = model(images, depths, attentions_choice=True)
    predictions = torch.argmax(logits, dim=1)

    ## averaging in block  4*[12, 4, 65, 65] -> [12, 4, 65, 65]
    attention_maps_img = torch.stack(attention_maps_img, dim=0).mean(dim=0)
    attention_maps_dpt = torch.stack(attention_maps_dpt, dim=0).mean(dim=0)
    
    attention_maps_img = process_attention(attention_maps_img, img_h, img_w)
    attention_maps_dpt = process_attention(attention_maps_dpt, img_h, img_w)


    show_images_with_attention(
        raw_images, raw_depths, attention_maps_img, attention_maps_dpt,
        label_ids, predictions, id_to_label, output, False
    )

def test(model, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, device="cuda", num_images = 16): 
    model.eval()

    # image_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.jpg')))
    # depth_paths = sorted(glob.glob(os.path.join(base_path, '*', '*.png')))
    image_paths = test_image_path
    depth_paths = test_depth_path
    labels, _ = getlabels_NYU(image_paths)

    indices = list(range(num_images))
    # indices = torch.randperm(len(image_paths))[:num_images] ## if random is not desired comment out
    
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
    logits, output_img, output_dpt, attention_maps_img, attention_maps_dpt = model(images, depths, attentions_choice=True)
    # print(f"attention_maps_img shape: {len(attention_maps_img)}, attention_maps_dpt shape: {len(attention_maps_dpt)}")
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

        output_name = f"attention_image\\attention_block{i}.png"

        show_images_with_attention(
            raw_images, raw_depths, attn_img, attn_dpt,
            label_ids, predictions, id_to_label, output_name, False, i
        )
    attention_data_final = []
    attention_data_final.extend(process_attention_data(images, depths, labels, attention_maps_img, attention_maps_dpt, len(attention_maps_img)-1, image_paths))
    rs, precission_topk = total_consistency(attention_data_final)
    print(f"total spearman rank correlation: {np.mean(rs):.4f}")

    output_rgbd = torch.cat((output_img, output_dpt), dim=1)  # Concatenate along feature dimension
    # mi = mi_regresser(output_img, output_rgbd)
    # print(f"Mutual Information: {mi:.4f}")

def batch_test_ViT(model, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device="cuda"):
    
    from train import Trainer
    from torch import nn, optim 
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay = 0.02)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        method=0,
        exp_name="RGB_ViT_infer",
        device=device
    )

    # Evaluateだけ実行
    accuracy, CLS_token, avg_loss, attention_data_final, wrong_images, correct_images = trainer.evaluate(test_loader, attentions_choice=True, infer_mode=True)

    print(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")


def batch_test_fusionViT(model, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device="cuda"):
    
    from train import Trainer
    from torch import nn, optim 

    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=None,
        loss_fn=loss_fn,
        method=2, 
        exp_name="share-fusion_infer",
        device=device
    )

    # Evaluateだけ実行
    accuracy, CLS_token, avg_loss, attention_data_final, wrong_images, correct_images = trainer.evaluate(test_loader, attentions_choice=True, infer_mode=True)
    
    mi_values = []
    with torch.no_grad():
        for i, (f_r, f_d) in enumerate(CLS_token):
            f_rgbd = torch.cat((f_r, f_d), dim=1)  
            mi = mi_regresser(f_r, f_rgbd)
            mi_values.append(mi.item())

    rs, precission_topk = total_consistency(attention_data_final)

    print(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")
    print(f"Spearman Rank Correlation: {np.mean(rs):.4f}")
    print(f"Precision at Top K: {np.mean(precission_topk):.4f}")
    print(f"Mutual Information: {np.mean(mi_values):.4f}")


def main(random_seed):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = r'..\data\nyu_data\nyu2'
    dataset_type = 1 # 0=WRGBD, 1=NYU, 2=TinyImageNet
    experiment_name = "vit-with-10-epochs"
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader

    transform1 = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    if dataset_type==0:
        load_datapath = load_datapath_WRGBD
    elif dataset_type==1:
        load_datapath = load_datapath_NYU
    elif dataset_type==2:
        load_datapath = load_datapath_TinyImageNet

    image_paths, depth_paths, labels =load_datapath(base_path, random_seed) 
    
    # max_num = 200
    # image_paths = image_paths[:max_num]
    # depth_paths = depth_paths[:max_num]
    # labels = labels[:max_num]

    _, _, test_loader, num_labels, label_mapping = get_dataloader(image_paths, depth_paths, batch_size, transform1, dataset_type)
    test_image_path = test_loader.dataset.image_paths
    test_depth_path = test_loader.dataset.depth_paths

    ## ----------------Visualize Attention Maps Toataly----------------

    config, model, _, _, _, label_mapping = load_experiment(
        experiment_name,
        checkpoint_name='onlyRGBViT_30epochs.pt.',
        depth=False,
        map_location=map_location
        )
    image_size = config['image_size']
    # RGB_visualize_attention_NYU(model, test_image_path, label_mapping, image_size, "attention_image\ViTattention.png", device=device)

    # config, model_latefusion, _, _, _, label_mapping = load_experiment(
    #     experiment_name,
    #     checkpoint_name='latefusion_30epochs.pt',
    #     depth=True,
    #     map_location=map_location
    #     )

    # RGBD_visualize_attention_NYU(model_latefusion, test_image_path, test_depth_path, label_mapping, image_size, "attention_image\latefusion_attention.png", device=device)

    config, model_sharefusion, _, _, _, label_mapping = load_experiment(
        experiment_name,
        checkpoint_name='sharefusion_30epoch.pt',
        depth=True,
        map_location=map_location
        )
    # image_size = config['image_size']
    # RGBD_visualize_attention_NYU(model_sharefusion, test_image_path, test_depth_path, label_mapping, image_size, "attention_image\sharefusion_attention.png", device=device)

    dim = config["hidden_size"]
    from module import CLUB
    mi_regresser = CLUB(dim, dim*2, 64).to(device)
    mi_regresser.load_state_dict(torch.load(r"experiments\vclub_model\NYU_0.8train_seed42_dim_48_96.pth", map_location=device, weights_only=True))
    mi_regresser.eval()

    ## ----------------Visualize Attention Maps per Block----------------

    # test(model_sharefusion, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, device="cuda", num_images = 15)


    ## ----------------Test Accracy, Spearman, Mutual Information----------------

    # batch_test_ViT(model, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device)
    
    # batch_test_fusionViT(model_latefusion, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device)

    batch_test_fusionViT(model_sharefusion, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device)

    


if __name__ == "__main__":
    random_seed = 42
    for i in range(5):
        print(f"Run in randomseed{random_seed}")
        main(random_seed)
        random_seed -= 1
    

