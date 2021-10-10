import numpy as np

def weighted_accuracy(y_test, y_pred):
    y_abs = np.abs(y_test)
    norm = y_abs.sum()
    score = ((np.sign(y_pred) == np.sign(y_test)) * y_abs).sum() / norm
    return score
