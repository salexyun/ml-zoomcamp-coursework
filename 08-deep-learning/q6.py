import numpy as np

from common import load_history

print("Q6. Mean of last 5 evaluation accuracies:", round(float(np.mean(load_history("augmented")["evaluation_accuracy"][-5:])), 2))
