import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VOCAB = {
    "<pad>": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "yellow": 4,
    "purple": 5,
    "orange": 6,
    "circle": 7,
    "square": 8,
    "triangle": 9,
    "diamond": 10,
    "moving": 11,
    "left": 12,
    "right": 13,
    "up": 14,
    "down": 15,
    "slow": 16,
    "fast": 17,
}

REV_VOCAB = {v: k for k, v in VOCAB.items()}

PROMPTS = [
    "red circle moving right",
    "blue square moving left",
    "green triangle moving up",
    "yellow diamond moving down",
    "purple circle moving right",
    "orange square moving left",
]

N_STEPS = 100
BETAS = torch.linspace(1e-4, 0.02, N_STEPS)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, dim=0)


def tokenize_prompt(prompt: str) -> List[int]:
    tokens = [VOCAB.get(token, VOCAB["<pad>"]) for token in prompt.lower().split()]
    return tokens


def parse_prompt(prompt: str) -> Tuple[str, str, str]:
    color = "red"
    shape = "circle"
    direction = "right"

    for key in ["red", "blue", "green", "yellow", "purple", "orange"]:
        if key in prompt:
            color = key
            break

    for key in ["circle", "square", "triangle", "diamond"]:
        if key in prompt:
            shape = key
            break

    for key in ["left", "right", "up", "down"]:
        if key in prompt:
            direction = key
            break

    return color, shape, direction


def render_shape(size: int, cx: float, cy: float, shape: str, strength: float = 1.0) -> np.ndarray:
    y_idx, x_idx = np.mgrid[0:size, 0:size]
    x_norm = x_idx - cx
    y_norm = y_idx - cy
    d = np.sqrt(x_norm ** 2 + y_norm ** 2)

    if shape == "circle":
        mask = d < size * 0.18
    elif shape == "square":
        mask = (np.abs(x_norm) < size * 0.2) & (np.abs(y_norm) < size * 0.2)
    elif shape == "triangle":
        x = x_norm / (size * 0.25)
        y = y_norm / (size * 0.25)
        mask = (y > -x - 1.0) & (y < x + 1.0) & (y < 1.0)
    elif shape == "diamond":
        mask = (np.abs(x_norm) + np.abs(y_norm)) < size * 0.22
    else:
        mask = d < size * 0.18

    frame = np.zeros((size, size), dtype=np.float32)
    frame[mask] = strength
    return frame


def generate_synthetic_video(prompt: str, frames: int = 8, size: int = 16):
    color, shape, direction = parse_prompt(prompt)
    object_strength = 1.0

    video = np.zeros((frames, size, size), dtype=np.float32)
    center_positions = np.linspace(2.0, size - 3.0, frames)

    for i in range(frames):
        cx = center_positions[i]
        cy = size / 2.0

        if direction == "left":
            cx = size - 2 - (size - 4) * (i / max(frames - 1, 1))
        elif direction == "right":
            cx = 2 + (size - 4) * (i / max(frames - 1, 1))
        elif direction == "up":
            cy = 2 + (size - 4) * (i / max(frames - 1, 1))
        elif direction == "down":
            cy = size - 2 - (size - 4) * (i / max(frames - 1, 1))

        frame = render_shape(size, cx, cy, shape, strength=object_strength)
        video[i] = frame

    return video


class PromptVideoDataset(Dataset):
    def __init__(self, prompts: List[str], frames: int = 8, size: int = 16):
        self.prompts = prompts
        self.frames = frames
        self.size = size

    def __len__(self):
        return len(self.prompts) * 32

    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]
        video = generate_synthetic_video(prompt, frames=self.frames, size=self.size)
        tokens = torch.tensor(tokenize_prompt(prompt), dtype=torch.long)
        return torch.from_numpy(video).float().unsqueeze(0), tokens


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    t = t.to(x0.device)
    sqrt_alpha_cumprod = torch.sqrt(ALPHAS_CUMPROD[t]).to(x0.device).view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - ALPHAS_CUMPROD[t]).to(x0.device).view(-1, 1, 1, 1, 1)
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.mix = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        residual = self.conv(self.norm(x))
        gamma, beta = self.mix(cond).chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1, 1)
        return x + residual * (1.0 + gamma) + beta


class VideoDiffusionModel(nn.Module):
    def __init__(self, vocab_size: int = len(VOCAB), text_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.text_encoder = nn.Embedding(vocab_size, text_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_conv = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            FiLMBlock(16, hidden_dim),
            FiLMBlock(16, hidden_dim),
            FiLMBlock(16, hidden_dim),
        ])
        self.output_conv = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, noisy_video, text_tokens, timestep):
        text_emb = self.text_encoder(text_tokens)
        text_emb = text_emb.mean(dim=1)
        cond = self.text_mlp(text_emb) + self.time_mlp(timestep.float().view(-1, 1))

        x = self.input_conv(noisy_video)
        for block in self.blocks:
            x = block(x, cond)
        x = self.output_conv(x)
        return x


def save_gif(video_frames, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames = []
    for frame in video_frames:
        data = np.clip(frame, 0, 1)
        gray = np.uint8(data * 255)
        img = Image.fromarray(gray, mode="L")
        frames.append(img)

    if len(frames) == 0:
        raise ValueError("No frames to save")

    frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=200)


def denoise_sample(model, prompt: str, frames: int = 8, size: int = 16, steps: int = 20):
    model.eval()
    tokens = torch.tensor([tokenize_prompt(prompt)], dtype=torch.long)
    x = torch.randn(1, 1, frames, size, size)

    with torch.no_grad():
        for step in reversed(range(0, N_STEPS, max(1, N_STEPS // steps))):
            t = torch.full((1,), step, dtype=torch.long)
            pred_noise = model(x, tokens, t)
            a_bar = ALPHAS_CUMPROD[step].to(x.device)
            pred_x0 = (x - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar)
            x = pred_x0

    x = x.clamp(0, 1)
    return x[0, 0].cpu().numpy()


def collate_batch(batch):
    videos = []
    token_lists = []
    max_len = max(len(tokens) for _, tokens in batch)

    for video, tokens in batch:
        videos.append(video.unsqueeze(0))
        pad_len = max_len - len(tokens)
        padded = F.pad(tokens, (0, pad_len), value=VOCAB["<pad>"])
        token_lists.append(padded)

    return torch.cat(videos, dim=0), torch.stack(token_lists)


if __name__ == "__main__":
    print("Synthetic data example:")
    sample_prompt = "red circle moving right"
    video = generate_synthetic_video(sample_prompt)
    print(video.shape)
    print(video.min(), video.max())
