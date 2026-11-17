import numpy as np

from common import load_history

print("Q3. Median train accuracy:", round(float(np.median(load_history("baseline")["train_accuracy"])), 2))
