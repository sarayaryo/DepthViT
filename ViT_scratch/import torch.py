import torch

device = torch.device("cuda")
print(f"Using device: {device}")
x = torch.randn(1024, 1024, 1024).to(device)  # 約4GB
y = torch.matmul(x, x)
print("Success!")
