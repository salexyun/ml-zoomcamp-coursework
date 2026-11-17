import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42
HERE = Path(__file__).parent
DATA = HERE / "data"
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
RESIZE = transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR)

evaluation_transform = transforms.Compose([RESIZE, transforms.ToTensor(), NORMALIZE])
train_transform_augmented = transforms.Compose([
    RESIZE,
    transforms.RandomRotation(50, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomResizedCrop(
        200, scale=(0.9, 1.0), ratio=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE,
])


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.hidden = nn.Linear(32 * 99 * 99, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.hidden(x))
        return self.output(x)


def make_loader(split, transform, seed=None):
    dataset = datasets.ImageFolder(DATA / split, transform=transform)
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=20, shuffle=seed is not None, num_workers=0, generator=generator)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over loader; trains if an optimizer is given. Returns (loss, accuracy) per example."""
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            labels = labels.float().unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def train_phase(model, train_loader, evaluation_loader, criterion, optimizer, epochs=10):
    history = {k: [] for k in ("train_loss", "train_accuracy", "evaluation_loss", "evaluation_accuracy")}
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, criterion, optimizer)
        evaluation = run_epoch(model, evaluation_loader, criterion)
        for k, v in zip(history, (*train, *evaluation)):
            history[k].append(v)
        print(f"epoch {epoch:2d}: train loss={train[0]:.4f} acc={train[1]:.4f} | eval loss={evaluation[0]:.4f} acc={evaluation[1]:.4f}")
    return history


def load_history(name):
    return json.loads((HERE / f"history_{name}.json").read_text())
