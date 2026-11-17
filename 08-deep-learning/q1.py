import torch
import torch.nn as nn

from common import HairModel

# The model ends in a single linear unit (a raw logit), so the loss must apply the sigmoid itself.
logits = HairModel()(torch.zeros(2, 3, 200, 200))
print("Output shape:", tuple(logits.shape))
print("Q1. Loss:", nn.BCEWithLogitsLoss.__name__)
