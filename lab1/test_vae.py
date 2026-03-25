import torch
import torch.nn as nn
import numpy as np

# ===== 路径 =====
BASE_DIR = "/mnt/sdb1/feiyang/datascience/lab1/Animals_with_Attributes2/Features/ResNet101/"
test_feat_file = BASE_DIR + "test_features.txt"

model_path = BASE_DIR + "vae_model.pth"
mean_path  = BASE_DIR + "vae_mean.npy"
std_path   = BASE_DIR + "vae_std.npy"

output_file = BASE_DIR + "test_vae_features.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== VAE结构（必须和训练完全一致！）=====
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # encoder
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # decoder（⚠️ 必须保留，否则加载失败）
        self.fc2 = nn.Linear(latent_dim, 1024)
        self.fc3 = nn.Linear(1024, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


# ===== 读取数据 =====
def load_features(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(list(map(float, line.strip().split())))
    return np.array(data, dtype=np.float32)


print("Loading test data...")
X = load_features(test_feat_file)

# ===== 标准化（必须和训练一致！）=====
print("Loading normalization stats...")
mean = np.load(mean_path)
std  = np.load(std_path)

X = (X - mean) / std

X_tensor = torch.tensor(X).to(device)

# ===== 加载模型 =====
print("Loading model...")
checkpoint = torch.load(model_path, map_location=device)

model = VAE(
    checkpoint['input_dim'],
    checkpoint['latent_dim']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ===== encode =====
print("Encoding...")
Z = []

batch_size = 128

with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]

        mu, _ = model.encode(batch)   # ⭐ 用 μ（确定性）
        Z.append(mu.cpu().numpy())

Z = np.vstack(Z)

print("Latent shape:", Z.shape)

# ===== 保存 =====
np.savetxt(output_file, Z, fmt="%.6f")

print("Saved to:", output_file)