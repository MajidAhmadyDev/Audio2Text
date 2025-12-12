#%%
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random

# =======================================================
# 1) Fake Dataset
# =======================================================
class DummyDataset(Dataset):
    def __init__(self, size=500):
        self.data = [(torch.randn(10), torch.randint(0, 2, (1,))) for _ in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y


# =======================================================
# 2) Simple Model
# =======================================================
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# =======================================================
# 3) Train Function
# =======================================================
def train_one_epoch(model, loader, loss_fn, optimizer, epoch):

    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.squeeze().to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- Log every 10 batches ----
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "step": epoch * len(loader) + batch_idx
            })

    avg_loss = total_loss / len(loader)
    wandb.log({"epoch_loss": avg_loss})

    return avg_loss


# =======================================================
# 4) MAIN
# =======================================================
if __name__ == "__main__":

    # Hyperparameters
    config = {
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 10
    }

    # Initialize WandB
    wandb.init(
        project="pytorch-wandb-demo",
        name="run_" + str(random.randint(1000,9999)),
        config=config
    )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Model, Loss, Optimizer
    model = SimpleNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Train Loop
    for epoch in range(config["epochs"]):
        avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, epoch)
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")

    wandb.finish()
