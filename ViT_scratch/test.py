import torch
import torch.nn.functional as F
from module import VariationalMI, total_consistency_CLS, total_consistency_patch, _auto_topn_from_mass
from utils import load_experiment
from data import ImageDepthDataset, load_datapath_NYU, getlabels_NYU
from torchvision import transforms

from train import spearman_rank_correlation, total_consistency, process_attention_data
from PIL import Image
from train import Trainer
from torch import nn, optim
import glob
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from print_alpha_beta import print_alpha_beta
from train_CMI import ConditionalCLUB, process_attention_data_forCMI

def apply_mass_filter(att_map, mass=0.8):
    """
    2次元のAttention Mapに対し、mass(0.0~1.0)に含まれない領域を0にする。
    
    Args:
        att_map (np.ndarray): (H, W) のAttention Map (正規化前でもOK)
        mass (float): 累積質量の閾値 (例: 0.8)
        
    Returns:
        masked_map (np.ndarray): フィルタリング後のMap
        binary_mask (np.ndarray): 採用された領域が1、それ以外0のマスク
    """
    # 1次元化してTop-N個数を計算
    flat_att = att_map.reshape(-1)
    top_n = _auto_topn_from_mass(flat_att, mass=mass)

    # 大きい順にソート
    sorted_vals = np.sort(flat_att)[::-1]
    threshold = sorted_vals[top_n - 1] # top_n個目の値
    
    # マスク作成 
    binary_mask = (att_map >= threshold).astype(np.float32)
    
    # フィルタリング適用 
    masked_map = att_map * binary_mask
    
    return masked_map, binary_mask

def visualize_mass_stepwise(raw_rgb, raw_depth, att_img_list, att_dpt_list, scores, output_path=None):
    """
    2行 x 11列 のグリッドを描画する関数
    Col 0: 元画像
    Col 1~10: 10% ~ 100% のAttention Overlay (青色背景あり)
    """
    num_steps = 10
    cols = num_steps + 1 # 1 (original) + 10 (steps)
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    
    # 画像間の隙間をなくす設定
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.99, top=0.92, bottom=0.02)
    
    # フォントサイズ設定
    FONT_SIZE_TITLE = 40  # Top N% 用
    FONT_SIZE_LABEL = 60  # RGB/Depth 用 (かなり大きく)
    FONT_SIZE_SCORE = 30  # スコア用
    
    # ==========================================
    # 1行目: RGB
    # ==========================================
    
    # --- Col 0: 元画像 ---
    ax = axes[0, 0]
    ax.imshow(raw_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(-0.1, 0.5, "RGB ", transform=ax.transAxes, 
            fontsize=FONT_SIZE_LABEL, va='center', ha='right', rotation=0)
    
    # --- Col 1~10: Attention Overlay ---
    for i in range(num_steps):
        mass_label = int((i + 1) * 10)
        ax = axes[0, i + 1]
        
        # 背景
        ax.imshow(raw_rgb)
        
        # Attention
        att = att_img_list[i]
        
        # 正規化 (show_images_with_attentionと同様の正規化を想定)
        if att.max() > 0:
            att_norm = att / att.max()
        else:
            att_norm = att
            
        ax.imshow(att_norm, cmap='jet', alpha=0.5)
        
        # タイトル (Top N%)
        ax.set_title(f"Top {mass_label}%", fontsize=FONT_SIZE_TITLE, pad=20)
        
        # 軸と枠線を消す
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ==========================================
    # 2行目: Depth
    # ==========================================
    
    # --- Col 0: 元画像 ---
    ax = axes[1, 0]
    ax.imshow(raw_depth, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # 左側に大きなラベル
    ax.text(-0.1, 0.5, "Depth", transform=ax.transAxes, 
            fontsize=FONT_SIZE_LABEL, va='center', ha='right', rotation=0)

    # --- Col 1~10: Attention Overlay ---
    for i in range(num_steps):
        ax = axes[1, i + 1]
        
        # 背景
        ax.imshow(raw_depth, cmap='gray')
        
        # Attention
        att = att_dpt_list[i]
        if att.max() > 0:
            att_norm = att / att.max()
        else:
            att_norm = att
            
        ax.imshow(att_norm, cmap='jet', alpha=0.5)

        # score = scores[i]
        # ax.set_title(f"S={score:.3f}", fontsize=FONT_SIZE_SCORE) 
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 保存時設定
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved visualization to {output_path}")
    
    # plt.show()

def process_attention(attention_maps, img_h, img_w):
    attention_maps = attention_maps[:, :, 0, 1:].mean(dim=1)
    size = int(math.sqrt(attention_maps.size(-1)))
    attention_maps = attention_maps.view(-1, size, size)
    attention_maps = F.interpolate(attention_maps.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(1)
    return attention_maps

def show_images_with_attention(raw_images, raw_depths, attention_maps_img, attention_maps_dpt, label_ids, predictions, id_to_label, output=None, show=False, score_s=None):

    num_images = len(raw_images)
    img_h, img_w = raw_images[0].shape[:2]

    ## figsize (640* 4, 480) == (16, 3)
    fig = plt.figure(figsize=(40, 30))

    if score_s is None:
        score_s = [0.0] * num_images
    
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
                f"gt: {gt} / pred: {pred}/ {score_s[i]:.2f}", 
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


def RGB_visualize_attention_NYU(model, test_image_path, label_mapping, image_size, output=None, test_loader=None, device="cuda"):
    from PIL import Image
    model.eval()

    image_paths = test_image_path
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
    # plt.show()

def RGBD_visualize_attention_NYU(model, test_image_path, test_depth_path, label_mapping, image_size, output=None, test_loader=None, device="cuda"):
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

    transform1 = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])
    depth_transform = transform1
    test_transform = transform1

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

    score_s = spearman_rank_correlation(attention_maps_img, attention_maps_dpt)

    show_images_with_attention(
        raw_images, raw_depths, attention_maps_img, attention_maps_dpt,
        label_ids, predictions, id_to_label, output, score_s=score_s
    )

def layer_attention(model, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, exp_name="", device="cuda", num_images = 16): 
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

    transform1 = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])

    images = torch.stack([transform1(Image.fromarray(img)) for img in raw_images]).to(device)
    depths = torch.stack([transform1(Image.fromarray(dpt)) for dpt in raw_depths]).to(device)

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

        output_name = f"attention_image\\{exp_name}_attention_block{i}.png"

        show_images_with_attention(
            raw_images, raw_depths, attn_img, attn_dpt,
            label_ids, predictions, id_to_label, output_name, False, score_s
        )
    attention_data_final = []
    attention_data_final.extend(process_attention_data(images, depths, labels, attention_maps_img, attention_maps_dpt, len(attention_maps_img)-1, image_paths))
    rs, precission_topk = total_consistency(attention_data_final)
    print(f"total spearman rank correlation: {np.mean(rs):.4f}")

    output_rgbd = torch.cat((output_img, output_dpt), dim=1)  # Concatenate along feature dimension
    # mi = mi_regresser(output_img, output_rgbd)
    # print(f"Mutual Information: {mi:.4f}")

def top_N_attention(model, image_paths, depth_paths, label_mapping, output=None, device="cuda"):
    model.eval()

    labels, _ = getlabels_NYU(image_paths)

    num_images = 1
    indices = [11]
    # indices = torch.randperm(len(image_paths))[:num_images] ## if random is not desired comment out
    
    raw_images = [np.asarray(Image.open(image_paths[i]).convert("RGB")) for i in indices]
    raw_depths = [np.asarray(Image.open(depth_paths[i])) for i in indices]
    
    label_ids = [labels[i] for i in indices]
    id_to_label = label_mapping 
    # print(label_mapping)

    transform1 = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])
    depth_transform = transform1
    test_transform = transform1

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

    attention_maps_img_mass = []
    attention_maps_dpt_mass = []
    spearman_scores = []

    for i in range(1,11,1):
        mass = i * 0.1
        att_map_img, _ = apply_mass_filter(attention_maps_img[0].cpu().detach().numpy(), mass=mass)
        att_map_dpt, _ = apply_mass_filter(attention_maps_dpt[0].cpu().detach().numpy(), mass=mass)
        attention_maps_img_mass.append(att_map_img)
        attention_maps_dpt_mass.append(att_map_dpt)
        # spearman_rank_correlation expects batch dimension [b, H, W]
        s_list = spearman_rank_correlation(att_map_img[None, ...], att_map_dpt[None, ...])
        spearman_scores.append(s_list[0])


    visualize_mass_stepwise(raw_images[0], raw_depths[0], attention_maps_img_mass, attention_maps_dpt_mass, spearman_scores, output_path=output)

    

def batch_test_ViT(model, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device="cuda"):
    
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
    
    # # MI
    # mi_values = []
    # with torch.no_grad():
    #     for i, (f_r, f_d) in enumerate(CLS_token):
    #         f_rgbd = torch.cat((f_r, f_d), dim=1)  
    #         mi = mi_regresser(f_r, f_rgbd)
    #         mi_values.append(mi.item())

    # CMI
    f_r, f_d, labels = process_attention_data_forCMI(attention_data_final, device)
    CMI_value = mi_regresser.calculate_cmi(f_r, f_d, labels)

    rs, precission_topk = total_consistency_patch(attention_data_final)
    rs2, precission_topk2 = total_consistency_CLS(attention_data_final)

    print(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")
    print(f"Spearman Rank Correlation(patch): {np.mean(rs):.4f}")
    print(f"Spearman Rank Correlation(CLS): {np.mean(rs2):.4f}")
    print(f"Precision at Top K: {np.mean(precission_topk):.4f}")
    print(f"Precision at Top K (CLS): {np.mean(precission_topk2):.4f}")
    # print(f"Mutual Information: {np.mean(mi_values):.4f}")
    print(f"Conditional Mutual Information: {CMI_value:.4f}")


def main(random_seed, infer_model=2):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_type = 1 # 0=WRGBD, 1=NYU, 2=TinyImageNet

    # infer_model = 1 # 0=vit, 1=late-fusion, 2=share-fusion, 3=AR-fusion

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader

    transform1 = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    if dataset_type==0:
        name = "WRGBD"
        load_datapath = load_datapath_WRGBD
        base_path = r'..\data\rgbd-dataset-10k'
    elif dataset_type==1:
        name = "NYU"
        load_datapath = load_datapath_NYU
        base_path = r'../data/nyu_data/nyu2'
    elif dataset_type==2:
        name = "TinyImageNet"
        load_datapath = load_datapath_TinyImageNet

    image_paths, depth_paths, labels =load_datapath(base_path, random_seed) 
    
    # max_num = 200
    # image_paths = image_paths[:max_num]
    # depth_paths = depth_paths[:max_num]

    _, _, test_loader, num_labels, label_mapping = get_dataloader(image_paths, depth_paths, batch_size, transform1, dataset_type)
    test_image_path = test_loader.dataset.image_paths
    test_depth_path = test_loader.dataset.depth_paths


    config,_ , _, _, _, _ = load_experiment(
    experiment_name="NYU_latefusion",
    checkpoint_name="model_final.pt",
    depth=True,
    map_location=map_location
    )
    dim = config["hidden_size"]

    ## Mutual Information Regressor Load
    # from module import CLUB
    # mi_regresser = CLUB(dim, dim*2, 64).to(device)
    # mi_regresser.load_state_dict(torch.load(r"experiments\vclub_model\NYU_0.8train_seed42_dim_48_96.pth", map_location=device, weights_only=True))
    # mi_regresser.eval()

    ## Conditional MI Regressor Load
    def CMI_model_load(model_path, device):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        config = checkpoint['config']
        
        mi_regresser = ConditionalCLUB(
            x_dim=config['x_dim'], 
            y_dim=config['y_dim'], 
            num_classes=config['num_classes'], 
            hidden_size=config['hidden_size']
        ).to(device)
        mi_regresser.load_state_dict(checkpoint['model_state_dict'])
        mi_regresser.eval()
        return mi_regresser
    

    scheduler = True
    scheduler = False


    ## ----------------Visualize Attention Maps Toataly----------------

    if infer_model==0:
        experiment_name = name + "_ViT"
        if scheduler: experiment_name += "_lr1e-3"
        config, model, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name="model_final.pt",
            depth=False,
            map_location=map_location
            )
        image_size = config['image_size']
        # RGB_visualize_attention_NYU(model, test_image_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", device=device)
        batch_test_ViT(model, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device)

    if infer_model==1:
        experiment_name = name + "_latefusion"
        if scheduler: experiment_name += "_lr1e-3"    
        config, model_latefusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name="model_final.pt",
            depth=True,
            map_location=map_location
            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # layer_attention(model_latefusion, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, experiment_name, device=device, num_images = 12)
        # RGBD_visualize_attention_NYU(model_latefusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", device=device)
        batch_test_fusionViT(model_latefusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)
        # top_N_attention(model_latefusion, test_image_path, test_depth_path, label_mapping, output=f"attention_image\{experiment_name}_topN_attention_seed{random_seed}.png", device=device)
        
    
    if infer_model==2.1:
        experiment_name = name + "_sharefusion_a0.0_b0.5"
        if scheduler: experiment_name += "_lr1e-3"
        config, model_sharefusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name="model_final.pt",
            depth=True,
            map_location=map_location
            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # RGBD_visualize_attention_NYU(model_sharefusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", test_loader, device=device)
        batch_test_fusionViT(model_sharefusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)

    if infer_model==2.2:
        experiment_name = name + "_sharefusion_a0.5_b0.0"
        if scheduler: experiment_name += "_lr1e-3"
        config, model_sharefusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name="model_final.pt",
            depth=True,
            map_location=map_location
            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # RGBD_visualize_attention_NYU(model_sharefusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", test_loader, device=device)
        batch_test_fusionViT(model_sharefusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)

    if infer_model==2:
        experiment_name = name + "_sharefusion_a0.5_b0.5"
        if scheduler: experiment_name += "_lr1e-3"
        config, model_sharefusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name="model_final.pt",
            depth=True,
            map_location=map_location,

            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # RGBD_visualize_attention_NYU(model_sharefusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", test_loader, device=device)
        # layer_attention(model_sharefusion, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, experiment_name, device=device, num_images = 12)
        batch_test_fusionViT(model_sharefusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)
        # top_N_attention(model_sharefusion, test_image_path, test_depth_path, label_mapping, output=f"attention_image\{experiment_name}_topN_attention_seed{random_seed}.png", device=device)
    
    if infer_model==2.3:
        experiment_name = name + "_sharefusion_a0.25_b0.25"
        if scheduler: experiment_name += "_lr1e-3"
        config, model_sharefusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name="model_final.pt",
            depth=True,
            map_location=map_location
            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # RGBD_visualize_attention_NYU(model_sharefusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", test_loader, device=device)
        batch_test_fusionViT(model_sharefusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)


    if infer_model==2.5:
        experiment_name = name + "_sharefusion_alearn_blearn"

        checkpoint_name="model_final.pt"

        if scheduler: experiment_name += "_lr1e-3_revise"
        config, model_sharefusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name=checkpoint_name,
            depth=True,
            map_location=map_location,
            override_config={"learnable_alpha_beta": True}
            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # print_alpha_beta(model_sharefusion)
        # RGBD_visualize_attention_NYU(model_sharefusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", test_loader, device=device)
        batch_test_fusionViT(model_sharefusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)
        # layer_attention(model_sharefusion, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, experiment_name,device=device, num_images = 12)


    if infer_model==3:
        experiment_name = name + "_ARfusion_alearn_blearn"

        if scheduler: checkpoint_name = "model_16.pt"
        else: checkpoint_name="model_final.pt"

        if scheduler: experiment_name += "_lr1e-3"
        config, model_ARfusion, _, _, _, label_mapping = load_experiment(
            experiment_name,
            checkpoint_name=checkpoint_name,
            depth=True,
            map_location=map_location,
            override_config={"learnable_alpha_beta": True}
            )
        image_size = config['image_size']
        CMI_model_path = os.path.join("vclub_model", f"{experiment_name}_CMI_dim_65.pth")
        CMI_regresser = CMI_model_load(CMI_model_path, device)

        # print(a)
        # print_alpha_beta(model_ARfusion)
        # RGBD_visualize_attention_NYU(model_ARfusion, test_image_path, test_depth_path, label_mapping, image_size, f"attention_image\{experiment_name}_attention_seed{random_seed}.png", test_loader, device=device)
        batch_test_fusionViT(model_ARfusion, batch_size, dataset_type, test_loader, CMI_regresser, label_mapping, device)
        # layer_attention(model_ARfusion, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, experiment_name, device=device, num_images = 12)
        # top_N_attention(model_ARfusion, test_image_path, test_depth_path, label_mapping, output=f"attention_image\{experiment_name}_topN_attention_seed{random_seed}.png", device=device)

    # ## ----------------Visualize Attention Maps per Block----------------

    # # test(model_sharefusion, test_image_path, test_depth_path, mi_regresser, label_mapping, image_size, device="cuda", num_images = 15)


    # ## ----------------Test Accracy, Spearman, Mutual Information----------------

    # # batch_test_ViT(model, batch_size, dataset_type, test_loader, mi_regresser, label_mapping, device)
    

if __name__ == "__main__":
    
    random_seed = 62
    for i in range(1):
        print(f"NYU Run in randomseed{random_seed}")
        # print("=== ViT ===")
        # main(random_seed, 0)
        print("=== Late Fusion ===")
        main(random_seed, 1)
        print("=== sharefusion_a0.0_b0.5 ===")
        main(random_seed, 2.1)
        print("=== sharefusion_a0.5_b0.0 ===")
        main(random_seed, 2.2)
        print("=== sharefusion_a0.5_b0.5 ===")
        main(random_seed, 2)
        print("=== Share Fusion_a0.25_b0.25 ===")
        main(random_seed, 2.3)
        print("=== Share Fusion (Learnable α, β) ===")
        main(random_seed, 2.5)
        print("=== Agreement Refined Fusion ===")
        main(random_seed, 3)
        random_seed -= 1
    

