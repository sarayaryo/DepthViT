import torch
import torch.nn as nn
import numpy as np

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
    
    import torch

def train_and_save_vmi(f_r, f_rgbd, save_path="vmi_model.pth", dim=48, epochs=500, lr=1e-3):
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

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
    
    import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B, 128, 1, 1]
            nn.Flatten(),             # [B, 128]
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        return self.net(x)

import numpy as np
import torch
from scipy.stats import spearmanr

def _to_numpy(x):
    """torch.Tensor / np.ndarray を np.ndarray に統一"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _extract_attention_view(att, view="patch_patch", head_reduce="mean"):
    """
    attention を「評価したい視点」に変換して返す。
    返り値は np.ndarray で shape は:
      - view="patch_patch": (B, N, N)  ※N=64
      - view="cls_patch"  : (B, N)     ※N=64
    入力 att は (B,H,T,T) または (H,T,T) のどちらでもOK。
    """
    att = _to_numpy(att)

    # (H,T,T) -> (1,H,T,T) に揃える
    if att.ndim == 3:
        att = att[None, ...]  # (1,H,T,T)
    assert att.ndim == 4, f"attention must be 4D (B,H,T,T) or 3D (H,T,T), got {att.shape}"

    B, H, T, _ = att.shape

    # 視点の抽出
    if view == "patch_patch":
        # patch↔patch (CLS除外)
        att = att[:, :, 1:, 1:]      # (B,H,64,64)
    elif view == "cls_patch":
        # CLS→patch (CLS行, patch列)
        att = att[:, :, 0, 1:]       # (B,H,64)
    else:
        raise ValueError(f"Unknown view={view}")

    # head方向の集約
    if head_reduce == "mean":
        att = att.mean(axis=1)       # (B,64,64) or (B,64)
    elif head_reduce == "none":
        # ヘッド別で評価したいなら使える
        pass
    else:
        raise ValueError(f"Unknown head_reduce={head_reduce}")

    return att

def spearman_rank_correlation(attention_img, attention_dpt, view="patch_patch"):
    """
    Spearmanをバッチごとに返す。
    入力は (B,H,T,T) でも (H,T,T) でもOK（np/torch両方OK）。
    view:
      - "patch_patch": patch↔patch（CLS除外）
      - "cls_patch"  : CLS→patch
    """
    img = _extract_attention_view(attention_img, view=view, head_reduce="mean")
    dpt = _extract_attention_view(attention_dpt, view=view, head_reduce="mean")

    assert img.shape == dpt.shape, f"shape mismatch: {img.shape} vs {dpt.shape}"

    rs_batch = []
    for x, y in zip(img, dpt):
        # x,y は (64,64) か (64,)
        coeff, _ = spearmanr(x.reshape(-1), y.reshape(-1))
        rs_batch.append(coeff)
    return rs_batch

def precision_top_k(attention_img, attention_dpt, k=1.0, view="patch_patch"):
    """
    top-k% の index 一致率（バッチごと）を返す。
    view は spearman と同じ。
    """
    img = _extract_attention_view(attention_img, view=view, head_reduce="mean")
    dpt = _extract_attention_view(attention_dpt, view=view, head_reduce="mean")

    assert img.shape == dpt.shape, f"shape mismatch: {img.shape} vs {dpt.shape}"

    precisions_batch = []
    for x, y in zip(img, dpt):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        top_k = max(1, int(len(x_flat) * k))

        x_idx = np.argsort(x_flat)[::-1][:top_k]
        y_idx = np.argsort(y_flat)[::-1][:top_k]

        intersection = len(set(x_idx) & set(y_idx))
        precisions_batch.append(intersection / top_k)

    return precisions_batch

def total_consistency(attention_data, k=1.0, view="patch_patch"):
    """
    attention_data: list of dict
      entry["attention_img"], entry["attention_dpt"] は (H,T,T) or (B,H,T,T) を想定
      ※あなたの現状は (H,65,65) が入っているはず
    view:
      - "patch_patch" : 旧 total_consistency と同じ（CLS除外のpatch↔patch）
      - "cls_patch"   : CLS視点（CLS→patch）
    """
    rs = []
    precisions = []

    for entry in attention_data:
        att_img = entry["attention_img"]
        att_dpt = entry["attention_dpt"]

        rs_batch = spearman_rank_correlation(att_img, att_dpt, view=view)
        if view == "patch_patch":
            precision_batch = precision_top_k(att_img, att_dpt, k=k, view=view)
        else:
            precision_batch, _, _ = precision_top_k_mass(
                att_img, att_dpt, mass=0.8, view=view
            )

        rs.extend(rs_batch)
        precisions.extend(precision_batch)

    return rs, precisions

# 互換ラッパ（名前を残したい場合）
def total_consistency_patch(attention_data, k=0.3):
    return total_consistency(attention_data, k=k, view="patch_patch")

def total_consistency_CLS(attention_data, k=0.3):
    return total_consistency(attention_data, k=k, view="cls_patch")

def _auto_topn_from_mass(att_vec, mass=0.8):
    """
    att_vec: (N,) の attention（CLS→patch など）
    mass: 例 0.8（累積80%）
    return: top_n（最小で1以上）
    """
    v = np.asarray(att_vec).reshape(-1)

    # 負やNaNがあると壊れるので保険（softmax後なら通常不要）
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, None)

    s = v.sum()
    if s <= 0:
        return 1  # 全部0なら仕方ないので1

    idx = np.argsort(v)[::-1]
    sorted_v = v[idx]
    cum = np.cumsum(sorted_v) / s

    top_n = int(np.searchsorted(cum, mass, side="left") + 1)
    return max(1, min(top_n, len(v)))


def precision_top_k_mass(attention_img, attention_dpt, mass=0.8, view="cls_patch"):
    """
    CLS→patch attention の累積質量 mass を満たす最小 top_n を各サンプルで決め、
    その top_n の index 一致率を返す。

    return:
      precisions_batch: list[float]
      ks_batch: list[float]  # 参考：実際のk(割合)
      topn_batch: list[int]  # 参考：実際のtop_n
    """
    img = _extract_attention_view(attention_img, view=view, head_reduce="mean")  # (B,64)
    dpt = _extract_attention_view(attention_dpt, view=view, head_reduce="mean")  # (B,64)
    assert img.shape == dpt.shape, f"shape mismatch: {img.shape} vs {dpt.shape}"

    precisions_batch = []
    ks_batch = []
    topn_batch = []

    for x, y in zip(img, dpt):
        # top_n は「片方だけ」で決めると偏るので、両方で決めて max を採用（頑健）
        n_x = _auto_topn_from_mass(x, mass=mass)
        n_y = _auto_topn_from_mass(y, mass=mass)
        top_n = max(n_x, n_y)

        x_idx = np.argsort(x)[::-1][:top_n]
        y_idx = np.argsort(y)[::-1][:top_n]

        inter = len(set(x_idx) & set(y_idx))
        precisions_batch.append(inter / top_n)

        topn_batch.append(top_n)
        ks_batch.append(top_n / len(x))

    return precisions_batch, ks_batch, topn_batch

