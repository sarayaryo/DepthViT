import torch
from module import FeatureExtractor, CLUB


def train_and_save_vclub(f_r, f_rgbd, save_path="vclub_model.pth", dim=48, epochs=500, lr=1e-4):
    device = f_r.device
    club = CLUB(dim, dim*2, 64).to(device)
    optimizer = torch.optim.Adam(club.parameters(), lr=lr)

    club.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = club.learning_loss(f_r, f_rgbd)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: CLUB loss = {loss.item():.4f}")

    torch.save(club.state_dict(), save_path)
    print(f"CLUB model saved to: {save_path}")
    return club

def extract_features_from_dataloader(rgb_encoder, depth_encoder, dataloader, device):
    rgb_encoder.eval()
    depth_encoder.eval()
    
    f_r_list, f_d_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)

            f_r = rgb_encoder(images)         # → [B, 128]
            f_d = depth_encoder(depths)       # → [B, 128]

            f_r_list.append(f_r)
            f_d_list.append(f_d)

    f_r_all = torch.cat(f_r_list, dim=0).to(device)
    f_d_all = torch.cat(f_d_list, dim=0).to(device)
    f_rgbd_all = torch.cat((f_r_all, f_d_all), dim=1)  # Concatenate along feature dimension
    return f_r_all, f_rgbd_all


def main():
    from train import Trainer
    from torch import nn, optim
    from torchvision import transforms
    from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader
    from utils import load_experiment
    base_path = r'..\data\nyu_data\nyu2'
    experiment_name = "vit-with-10-epochs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## ---------- data preparation ----------

    # method: 0=Simple, 1=Early, 2=Late

    transform1 = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    dataset_type = 1  # 0: WRGBD, 1: NYU, 2: TinyImageNet
    if dataset_type==0:
        load_datapath = load_datapath_WRGBD
    elif dataset_type==1:
        load_datapath = load_datapath_NYU
    elif dataset_type==2:
        load_datapath = load_datapath_TinyImageNet

    image_paths, depth_paths, labels =load_datapath(base_path)
        
    # max_num = 400
    # image_paths = image_paths[:max_num]
    # depth_paths = depth_paths[:max_num]
    # labels = labels[:max_num]

    batch_size = 16
    train_loader, _, _, _, _ = get_dataloader(image_paths, depth_paths, batch_size, transform1, dataset_type)
    print(f"Number of training RGB and Depth: {len(train_loader.dataset)}")

    rgb_encoder = FeatureExtractor(input_channels=3, feature_dim=48).to(device)
    depth_encoder = FeatureExtractor(input_channels=1, feature_dim=48).to(device)

    f_r_all, f_rgbd_all = extract_features_from_dataloader(rgb_encoder, depth_encoder, train_loader, device)

    print(f"f_r_all shape: {f_r_all.shape}, f_rgbd_all shape: {f_rgbd_all.shape}")

    import os
    if not os.path.exists("experiments\\vclub_model"):
        os.makedirs("experiments\\vclub_model", exist_ok=True)
    
    ## ---------- train CLUB model ----------

    dim=f_r_all.shape[1]
    print(f"Feature dimension: {dim}")
    train_and_save_vclub(f_r_all, f_rgbd_all, save_path=r"experiments\vclub_model\NYU_0.8train_seed42_dim_48_96.pth", dim=f_r_all.shape[1], epochs=500, lr=1e-5)