from common import HairModel

model = HairModel()
for name, p in model.named_parameters():
    print(f"{name:15s} {p.numel():>10,}")
print("Q2. Total parameters:", sum(p.numel() for p in model.parameters()))
