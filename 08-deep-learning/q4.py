import numpy as np

from common import load_history

print("Q4. Std of train loss:", round(float(np.std(load_history("baseline")["train_loss"], ddof=0)), 3))
