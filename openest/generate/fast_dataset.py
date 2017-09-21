import functools
import xarray as xr
import numpy as np

class FastDataset(xr.Dataset):
    def __init__(self, data_vars, coords={}, attrs=None):
        # Do not call __init__ on Dataset, to avoid time cost
        self.original_data_vars = data_vars
        self.original_coords = coords

        self._variables = {name: FastDataArray(data_vars[name][1], data_vars[name][0] if isinstance(data_vars[name][0], tuple) else (data_vars[name][0],)) for name in data_vars}
        self._variables.update({name: FastDataArray(coords[name], (name,)) for name in coords})
        self._dims = {key: getattr(self._variables[key]._data, 'shape', ()) for key in self._variables}
        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
            
    def __str__(self):
        return str(xr.Dataset(self.original_data_vars, self.original_coords, self.attrs))

    def sum(self):
        return self.reduce(np.sum)

    def mean(self):
        return self.reduce(np.mean)

    def reduce(self, func):
        newvars = {}
        for key in self._variables:
            if key in self.original_coords:
                continue
            newvars[key] = ((), np.array(func(self._variables[key]._data)))
        return FastDataset(newvars, self.attrs)

    def transform(self, func):
        newvars = {}
        for key in self._variables:
            if key in self.original_coords:
                continue
            newvars[key] = (self._variables[key].original_coords, np.array(func(self._variables[key]._data)))
        return FastDataset(newvars, self.attrs)        

    def subset(self, names):
        newvars = {}
        for key in self._variables:
            if key in self.original_coords or key in names:
                continue
            newvars[key] = (self._variables[key].original_coords, self._variables[key]._data)
        return FastDataset(newvars, self.attrs)
    
    def __getitem__(self, name):
        return self._variables[name]
    
    def __getattr__(self, name):
        return self._variables[name]

class FastDataArray(xr.DataArray):
    def __init__(self, data, coords):
        # Do not call __init__ on DataArray, to avoid time cost
        self.original_coords = coords
        self._values = self._data = data

    def __len__(self):
        return len(self._values)

    def __str__(self):
        return str(xr.DataArray(self._data))

    @property
    def values(self):
        return self._values
    
    @property
    def _variable(self):
        # We don't want this layer: if returning a numpy doesn't work, time to add a new function
        return self._values

    @staticmethod
    def _binary_op(f, reflexive=False, join=None, **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            other_values = getattr(other, '_values', other)

            values = f(self._values, other_values)

            return FastDataArray(values, self.original_coords)
        
        return func

    def reduce(self, func, dim=None, axis=None, keep_attrs=False, **kwargs):
        data = func(self._values)
        # Only handles no reduction or total reduction
        newshape = getattr(data, 'shape', ())
        if newshape == ():
            return FastDataArray(data, ())
        elif newshape == self.shape:
            return FastDataArray(data, self.original_coords)
        else:
            if dim is not None:
                axis = self.original_coords.index(dim)
            newcoords = list(self.original_coords)
            del newcoords[axis]
            return FastDataArray(data, newcoords)

    def __array__(self):
        return np.asarray(self._values)

def region_groupby(ds, year, regions, region_indices):
    timevar = ds.time
    for ii in range(len(regions)):
        region = regions[ii]
            
        newvars = {}
        for var in ds:
            if var in ['time', 'region']:
                continue
            dsdata = ds._variables[var]._data
            if len(dsdata.shape) < 2:
                continue
                
            newvars[var] = (['time'], dsdata[:, region_indices[region]])
        subds = FastDataset(newvars, coords={'time': timevar}, attrs={'year': year})
            
        yield region, subds

FastDataArray.__array_priority__ = 80
xr.core.ops.inject_binary_ops(FastDataArray)
    
if __name__ == '__main__':
    ds = FastDataset({'x': ('time', np.ones(3))}, {'time': np.arange(3)})
    print ds
    print ds.x
    print ds.x * 3
