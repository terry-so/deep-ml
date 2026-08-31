import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:

    return((np.exp(scores-np.max(scores)))/np.sum(np.exp(scores-np.max(scores))))
    pass