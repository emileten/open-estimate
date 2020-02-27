import numpy as np
from numpy import linalg

def transform(predcount, bins, dropbin, totals):
    """
    Construct a transform matrix from an old set of predictors to a bin
    swapped set.

    The intercept is assumed to be the first predictor.

    Args:
        predcount (int): the number predictors, including the intercept.
        bins (list[int]): the indices of the bins amongst the predictors.
        dropbin (int): an index into `bins`
        totals (float): the value that all bin values would sum to,
            e.g. `1` if the bins are indicators
            e.g., 365 if the bins are daily over a year
    """
    
    transform = np.identity(predcount)
    transform[0, bins[dropbin]] = totals
    transform[bins, bins[dropbin]] = -1

    return transform

def swap_vcv(V, T):
    return linalg.inv(T).dot(V).dot(linalg.inv(T).transpose())

def swap_beta(beta, T):
    beta = np.array(beta)
    if beta.shape[0] == 1:
        beta = beta.transpose()
    return linalg.inv(T).dot(beta)

if __name__ == '__main__':
    ## An example with 3 bins summed over days within weeks
    X = np.array([[1, 2, 3, .5], [1, 2, 2, .4]])
    print("X =")
    print(X)
    T = transform(4, [1, 2], 1, 7)
    print("T =")
    print(T)
    print("Y =")
    print(X.dot(T))

    V = np.diag([1, .2, .2, .5])
    print("V =")
    print(V)
    print("W =")
    print(swap_vcv(V, T))

    beta = [1, 2, 3, 4]
    print("beta =")
    print(beta)
    print("gamma = ")
    print(swap_beta(beta, T))
