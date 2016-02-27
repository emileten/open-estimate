import numpy as np
import transform
from ..models.spline_model import SplineModel

def spline_find_bins(model, beta, vcv):
    bins = []
    for x in model.get_xx():
        mean = model.get_mean(x)
        sdev = model.get_sdev(x)

        mean_errors = (np.array(beta) - mean) ** 2
        sdev_errors = (np.array(np.sqrt(vcv.diag())) - sdev) ** 2
        errors = mean_errors + sdev_errors
        
        index = np.argmin(errors)
        if errors[index] < 1e-6 * mean:
            bins.append(index)
        else:
            raise LookupError("Cannot find " + str(mean) + " +- " + str(sdev))

    return bins

def spline_swap(model, beta, vcv, dropbin, totals):
    # Find the bins in beta and vcv
    bins = find_bins(model, beta, vcv)
    # Transform the entire beta and vcv matrix
    T = transform(len(beta), bins, dropbin, totals)
    
    vcv2 = swap_vcv(vcv, T)
    beta2 = swap_beta(beta, T)

    # Construct a new model
    xxs = {model.get_xx()[ii]: (beta2[bins[ii]], np.sqrt(vcv2[bins[ii], bins[ii]])) for ii in range(len(bins))}
    SplineModel.create_gaussian(xxs, model.xx_is_categorical)
