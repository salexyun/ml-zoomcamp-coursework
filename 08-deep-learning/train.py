"""Trains the baseline then augmented phase and writes history_baseline.json and history_augmented.json."""
import json

import torch
import torch.nn as nn

from common import HERE, SEED, HairModel, evaluation_transform, make_loader, seed_everything, train_phase, train_transform_augmented

seed_everything()
torch.set_num_threads(2)  # matches reference_train.py

evaluation_loader = make_loader("test", evaluation_transform)
model = HairModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

phases = {"baseline": (evaluation_transform, SEED), "augmented": (train_transform_augmented, SEED + 1)}
for name, (transform, seed) in phases.items():
    print(f"--- {name} ---")
    history = train_phase(model, make_loader("train", transform, seed), evaluation_loader, criterion, optimizer)
    (HERE / f"history_{name}.json").write_text(json.dumps(history, indent=2) + "\n")
