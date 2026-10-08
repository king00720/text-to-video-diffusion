import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import (
    PROMPTS,
    PromptVideoDataset,
    VideoDiffusionModel,
    collate_batch,
    q_sample,
)


def train(args):
    dataset = PromptVideoDataset(PROMPTS, frames=8, size=16)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    model = VideoDiffusionModel()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, (video_batch, token_batch) in enumerate(dataloader):
            batch_size = video_batch.size(0)
            t = torch.randint(0, 100, (batch_size,))
            noise = torch.randn_like(video_batch)
            noisy_video = q_sample(video_batch, t, noise)
            pred_noise = model(noisy_video, token_batch, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs} | loss: {total_loss / max(1, len(dataloader)):.6f}")

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pt")

    torch.save(model.state_dict(), "checkpoints/final_model.pt")
    print("Training complete. Model saved to checkpoints/final_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the toy text-to-video diffusion model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)
