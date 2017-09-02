class FastDataset(xr.Dataset):
    def __init__(self, data_vars, coords):
        # Do not call __init__ on Dataset, to avoid time cost
        self.data_vars = data_vars
        self.coords = coords

        self._variables = {name: data_vars[name][1] for name in data_vars}

    def __item__(self, name):
        return self._variables[name]

class FastDataArray(xr.DataArray):
    def __init__(self, data):
        # Do not call __init__ on DataArray, to avoid time cost
        self.values = self._values = self.data = data
