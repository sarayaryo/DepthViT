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
