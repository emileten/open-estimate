import sys, re
from scipy.io import loadmat
import numpy as np

#filename = "/Users/jrising/Dropbox/James-Sol/DATA/countries/dateline_masks"
filename = "/Users/jrising/Dropbox/James-Sol/DATA/country_masks_w_monthly_mean_temp.mat"
# filename = sys.argv[1]
data = loadmat(filename)
for key in data:
    if not re.match(r"^__(.+)__$", key):
        if isinstance(data[key], np.ndarray):
            print(key, data[key].dtype, data[key].shape)
            if data[key].dtype.names is not None and len(data[key].dtype.names) > 0:
                for name in data[key].dtype.names:
                    print(name, data[key][name][0, 0].dtype, data[key][name][0, 0].shape)
        else:
            print(key, data[key].__class__.__name__)

def extract_labeled_map(struct, ii, name_key, data_key, xs_key, ys_key):
    name = struct[name_key][0, 0][ii][0][0]
    return Map(name, struct[data_key][0, 0][:,:,ii], struct[xs_key][0, 0], struct[ys_key][0, 0])

class Map:
    def __init__(self, name, data, xs, ys):
        self.name = name
        self.data = data
        self.xs = xs
        self.ys = ys

m1 = extract_labeled_map(data['dateline_masks_density_10'], 1, 'iso', 'masks', 'lon', 'lat')
