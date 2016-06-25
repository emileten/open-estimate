import numpy as np
import transform
from ..models.spline_model import SplineModel
from ..models.bin_model import BinModel

def swap_any(model, beta, vcv, dropbin, totals):
    if isinstance(model, SplineModel):
        return swap_spline(model, beta, vcv, dropbin, totals)
    if isinstance(model, BinModel) and isinstance(model.model, SplineModel):
        return swap_bin(model, beta, vcv, dropbin, totals)

    raise NotImplementedError("Do not know how to swap bin for " + str(model))

def find_bins(means, sdevs, beta, vcv):
    bins = []
    for ii in range(len(means)):
        if np.isnan(means[ii]):
            continue

        mean_errors = (np.array(beta) - means[ii]) ** 2
        sdev_errors = (np.sqrt(np.array(vcv).diagonal()) - sdevs[ii]) ** 2
        errors = mean_errors + sdev_errors

        index = np.argmin(errors)
        if errors[index] < 1e-6 * np.abs(means[ii]):
            bins.append(index)
        else:
            raise LookupError("Cannot find " + str(means[ii]) + " +- " + str(sdevs[ii]))

    return bins

def swap_spline(model, beta, vcv, dropbin, totals):
    means = [model.get_mean(x) for x in model.get_xx()]
    sdevs = [model.get_sdev(x) for x in model.get_xx()]
    means2, sdevs2 = swap_values(means, sdevs, beta, vcv, dropbin, totals)

    # Construct a new model
    jj = 0 # index into xx (goes 1 ahead at dropped bin)
    xxs = {}
    order = []
    for ii in range(len(means)):
        x = model.get_xx()[ii]
        order.append(x)
        if ii == dropbin:
            xxs[x] = (np.nan, np.nan)
        else:
            xxs[x] = (means2[jj], sdevs2[jj] ** 2)
            jj += 1

    return SplineModel.create_gaussian(xxs, order=order, xx_is_categorical=model.xx_is_categorical)

def swap_bin(model, beta, vcv, dropbin, totals):
    model2 = swap_spline(model.model, beta, vcv, dropbin, totals)
    return BinModel(model.get_xx(), model2)

def swap_values(means, sdevs, beta, vcv, dropbin, totals):
    # Find the bins in beta and vcv
    bins = find_bins(means, sdevs, beta, vcv)
    # Transform the entire beta and vcv matrix
    T = transform.transform(len(beta), bins, dropbin, totals)
    
    vcv2 = transform.swap_vcv(vcv, T)
    beta2 = transform.swap_beta(beta, T)

    means2 = [beta2[ii] for ii in bins]
    sdevs2 = [np.sqrt(vcv2[ii, ii]) for ii in bins]

    return means2, sdevs2
