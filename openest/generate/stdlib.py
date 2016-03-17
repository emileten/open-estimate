import numpy as np
from daily import *
from functions import *

def logscalefunc(x, s):
    return s * (np.exp(x) - 1)
