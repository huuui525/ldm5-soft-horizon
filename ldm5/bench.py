"""Quick benchmark: isolate I/O vs GPU compute."""
import time
import torch
from .config import default_config
from .models import build_autoencoder

cfg = default_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
print("cudnn:", torch.backends.cudnn.version(), "enabled:", torch.backends.cudnn.enabled)

model = build_autoencoder(cfg.ae).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(8, 5, 224, 224, device=device)

# Warmup
for _ in range(3):
    posterior = model.encode(x).latent_dist
    z = posterior.sample()
    recon = model.decode(z).sample
    loss = (recon - x).abs().mean() + 1e-6 * posterior.kl().mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
torch.cuda.synchronize() if device == "cuda" else None

# Time
t0 = time.time()
N = 10
for _ in range(N):
    posterior = model.encode(x).latent_dist
    z = posterior.sample()
    recon = model.decode(z).sample
    loss = (recon - x).abs().mean() + 1e-6 * posterior.kl().mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
torch.cuda.synchronize() if device == "cuda" else None
print(f"Avg step: {(time.time()-t0)/N*1000:.1f} ms  (batch=8, 5x224x224)")
