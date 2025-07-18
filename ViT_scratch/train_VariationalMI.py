import torch
import torch.nn as nn
import math
from module import VariationalMI

def train_and_save_vmi(f_r, f_rgbd, save_path="vmi_model.pth", dim=48, epochs=500, lr=1e-4):
    # dim = f_r.shape[2]
    model = VariationalMI(dim).to(f_r.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mi = model(f_r, f_rgbd)
        loss = -mi  # 相互情報量を最大化
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: MI = {mi.item():.4f}")

    # 保存
    torch.save(model.state_dict(), save_path)
    print(f"VMI model saved to: {save_path}")
    return model



def main():
    from train import Trainer
    import torch
    from torch import nn, optim
    from torchvision import transforms
    from train import Trainer
    from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader
    from utils import load_experiment
    base_path = r'..\data\nyu_data\nyu2'
    experiment_name = "vit-with-10-epochs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, model_sharefusion, _, _, _, label_mapping = load_experiment(
        experiment_name,
        checkpoint_name='latefusion_30epochs.pt',
        depth=True,
        map_location = device
        )
    optimizer = optim.AdamW(model_sharefusion.parameters(), lr=1e-2, weight_decay = 0.02)
    loss_fn = nn.CrossEntropyLoss()
    # method: 0=Simple, 1=Early, 2=Late
    trainer = Trainer(
        model=model_sharefusion,
        optimizer=optimizer,
        loss_fn=loss_fn,
        method=2, 
        exp_name="share-fusion_infer",
        device =device,
    )
    transform1 = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

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
    train_loader, valid_loader, test_loader, num_labels, label_mapping = get_dataloader(image_paths, depth_paths, batch_size, transform1, dataset_type)
    accuracy, CLS_tokens, avg_loss, attention_data_final, wrong_images, correct_images = trainer.evaluate(train_loader, attentions_choice=True, infer_mode=True)

    f_r_all = torch.cat([pair[0] for pair in CLS_tokens], dim=0)      # shape: [total_samples, dim]
    f_rgbd_all = torch.cat([pair[1] for pair in CLS_tokens], dim=0)   # shape: [total_samples, dim]

    print(f"f_r_all shape: {f_r_all.shape}, f_rgbd_all shape: {f_rgbd_all.shape}")

    import os
    if not os.path.exists("experiments\\vmi_model"):
        os.makedirs("experiments\\vmi_model", exist_ok=True)

    train_and_save_vmi(f_r_all, f_rgbd_all, save_path=r"experiments\vmi_model\late30.pth", dim=f_r_all.shape[1], epochs=500, lr=1e-5)
    

if __name__ == "__main__":
    main()