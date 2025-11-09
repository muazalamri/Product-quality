import torch.nn as nn

Model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),   # [B, 1, 128, 128] → [B, 16, 128, 128]
    nn.ReLU(),
    nn.MaxPool2d(2),                  # → [B, 16, 64, 64]

    nn.Conv2d(16, 32, 3, padding=1),  # → [B, 32, 64, 64]
    nn.ReLU(),
    nn.MaxPool2d(2),                  # → [B, 32, 32, 32]

    nn.Conv2d(32, 64, 3, padding=1),  # → [B, 64, 32, 32]
    nn.ReLU(),
    nn.MaxPool2d(2),                  # → [B, 64, 16, 16]

    nn.Conv2d(64, 128, 3, padding=1), # → [B, 128, 16, 16]
    nn.ReLU(),
    nn.MaxPool2d(2),                  # → [B, 128, 8, 8]

    nn.Flatten(1),                     # → [B, 128*8*8 = 8192]
    nn.Linear(8192, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
