import numpy as np

from common import load_history

print("Q5. Mean evaluation loss:", round(float(np.mean(load_history("augmented")["evaluation_loss"])), 3))
