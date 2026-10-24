import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_gpt import RocketDecisionTransformer

class RecoveryTrajectoryDataset(Dataset):
    def __init__(self, data_path, context_len=20):
        with open(data_path, "rb") as f:
            self.trajectories = pickle.load(f)
        self.context_len = context_len

        # Filter high-score successful recovery flights
        self.trajectories = [t for t in self.trajectories if t["total_return"] > 0]
        print(f"[+] Loaded {len(self.trajectories)} successful recovery flights into dataset.", flush=True)

    def __len__(self):
        return len(self.trajectories) * 40

    def __getitem__(self, idx):
        traj = self.trajectories[idx % len(self.trajectories)]
        T = len(traj["observations"])

        if T > self.context_len:
            start = np.random.randint(0, T - self.context_len)
            end = start + self.context_len
            states = traj["observations"][start:end]
            actions = traj["actions"][start:end]
            rtg = traj["returns_to_go"][start:end, None]
            timesteps = traj["timesteps"][start:end]
        else:
            pad_len = self.context_len - T
            states = np.pad(traj["observations"], ((0, pad_len), (0, 0)), mode="edge")
            actions = np.pad(traj["actions"], ((0, pad_len), (0, 0)), mode="constant")
            rtg = np.pad(traj["returns_to_go"][:, None], ((0, pad_len), (0, 0)), mode="edge")
            timesteps = np.pad(traj["timesteps"], (0, pad_len), mode="edge")

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rtg, dtype=torch.float32),
            torch.tensor(timesteps, dtype=torch.long)
        )

def train_gpt(epochs=20, batch_size=32, lr=8e-4, save_dir="models"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "rocket_trajectories.pkl")
    save_path = os.path.join(script_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)

    dataset = RecoveryTrajectoryDataset(data_path, context_len=20)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cpu")
    model = RocketDecisionTransformer(
        state_dim=7,
        act_dim=2,
        d_model=64,
        nhead=4,
        num_layers=3,
        context_len=20
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=== Training Rocket GPT (Decision Transformer) on CPU ===", flush=True)
    print(f"Parameters: {total_params:,} | Context Window: 20 timesteps | Epochs: {epochs}", flush=True)
    print("-" * 65, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    criterion = nn.MSELoss()

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for states, actions, rtg, timesteps in loader:
            states = states.to(device)
            actions = actions.to(device)
            rtg = rtg.to(device)
            timesteps = timesteps.to(device)

            optimizer.zero_grad()
            preds = model(states, actions, rtg, timesteps)
            loss = criterion(preds, actions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - start_time
        accuracy_pct = max(0.0, min(100.0, (1.0 - np.sqrt(avg_loss)) * 100.0))

        print(f"Epoch {epoch:2d}/{epochs} | Loss (MSE): {avg_loss:.5f} | Fit Accuracy: {accuracy_pct:.1f}% | Time: {elapsed:.1f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_file = os.path.join(save_path, "rocket_gpt.pt")
            torch.save(model.state_dict(), model_file)

    print(f"\n[+] Rocket GPT reached peak fit ({best_loss:.5f} MSE)! Saved to {model_file}!", flush=True)

if __name__ == "__main__":
    train_gpt()
