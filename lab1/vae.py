import torch
import torch.nn as nn
import numpy as np

# ===== 路径 =====
BASE_DIR = "/mnt/sdb1/feiyang/datascience/lab1/Animals_with_Attributes2/Features/ResNet101/"
train_feat_file = BASE_DIR + "train_features.txt"
output_file = BASE_DIR + "train_vae_features.txt"

model_path = BASE_DIR + "vae_model.pth"
mean_path = BASE_DIR + "vae_mean.npy"
std_path  = BASE_DIR + "vae_std.npy"

# ===== 超参数 =====
input_dim = 2048
latent_dim = 512
batch_size = 128
epochs = 200
lr = 1e-3
beta = 0.1   # ⭐ 关键（比标准VAE更适合你）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 读取数据 =====
def load_features(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(list(map(float, line.strip().split())))
    return np.array(data, dtype=np.float32)


print("Loading data...")
X = load_features(train_feat_file)

# ===== 标准化（非常重要）=====
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X = (X - mean) / std

# 保存标准化参数（test 必须用）
np.save(mean_path, mean)
np.save(std_path, std)

print("Data shape:", X.shape)

# 转 tensor
X_tensor = torch.tensor(X).to(device)

dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ===== VAE =====
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.fc2 = nn.Linear(latent_dim, 1024)
        self.fc3 = nn.Linear(1024, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


model = VAE(input_dim, latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===== 自适应学习率 =====
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-5,
    verbose=True
)


# ===== 训练 =====
print("Training VAE...")
for epoch in range(epochs):
    total_loss = 0

    for (x,) in loader:
        x_hat, mu, logvar = model(x)

        recon_loss = ((x - x_hat) ** 2).mean()
        kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))

        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()

        # ⭐ 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()

    scheduler.step(total_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")


# ===== 提取特征（用 μ）=====
print("Extracting latent features...")
model.eval()
Z = []

with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]
        mu, _ = model.encode(batch)
        Z.append(mu.cpu().numpy())

Z = np.vstack(Z)

print("Latent shape:", Z.shape)


# ===== 保存特征 =====
np.savetxt(output_file, Z, fmt="%.6f")
print("Saved features to:", output_file)


# ===== 保存模型 =====
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'latent_dim': latent_dim
}, model_path)

print("Model saved to:", model_path)