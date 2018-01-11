import copy, functools
import xarray as xr
import numpy as np

def as_name(coord):
    assert isinstance(coord, str), "Not a string: %s" % coord
    return coord

class FastDataset(xr.Dataset):
    def __init__(self, data_vars, coords={}, attrs=None):
        # Do not call __init__ on Dataset, to avoid time cost
        self.original_data_vars = data_vars
        self.original_coords = coords

        self._variables = {}
        for name in data_vars:
            if isinstance(data_vars[name], tuple):
                self._variables[name] = FastDataArray(data_vars[name][1], tuple(map(as_name, data_vars[name][0] if isinstance(data_vars[name][0], tuple) or isinstance(data_vars[name][0], list) else (data_vars[name][0],))), self)
            elif isinstance(data_vars[name], FastDataArray):
                self._variables[name] = data_vars[name]
            elif name not in coords:
                self._variables[name] = FastDataArray(data_vars[name]._data, data_vars[name].dims, self)
        self._variables.update({name: FastDataArray(coords[name], (name,), self) for name in coords})
        self._dims = {key: getattr(self._variables[key]._data, 'shape', ()) for key in self._variables}
        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
            
    def __str__(self):
        result = "FastDataset: [%s] x [%s]" % (', '.join(self.original_data_vars.keys()), ', '.join(self.original_coords.keys()))
        for key in self.attrs:
            result += "\n\t%s: %s" % (key, str(self.attrs[key]))

        return result

    def sum(self, dim=None):
        return self.reduce(np.sum, dim=dim)

    def mean(self, dim=None):
        return self.reduce(np.mean, dim=dim)

    def reduce(self, func, dim=None):
        if dim is not None:
            assert isinstance(dim, str)
            
        newvars = {}
        for key in self._variables:
            if key in self.original_coords:
                continue
            if dim is not None and dim in self._variables[key].original_coords:
                axis = self._variables[key].original_coords.index(dim)
                newvars[key] = ((), np.array(func(self._variables[key]._data, axis=axis)))
            else:
                newvars[key] = ((), np.array(func(self._variables[key]._data)))

        if dim is None:
            return FastDataset(newvars, {}, self.attrs)
        else:
            newcoords = copy.copy(self.original_coords)
            del newcoords[dim]
            return FastDataset(newvars, newcoords, self.attrs)

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
            if key in self.original_coords or key not in names:
                continue
            newvars[key] = (self._variables[key].original_coords, self._variables[key]._data)
        return FastDataset(newvars, self.original_coords, self.attrs)

    def rename(self, name_dict):
        # Cannot rename coordinates
        newvars = {}
        for key in self._variables:
            if key in self.original_coords:
                continue
            if key in name_dict:
                newvars[name_dict[key]] = (self._variables[key].original_coords, self._variables[key]._data)
            else:
                newvars[key] = (self._variables[key].original_coords, self._variables[key]._data)
        return FastDataset(newvars, self.original_coords, self.attrs)        

    def sel(self, **kwargs):
        newcoords = {key: self.original_coords[key] for key in self.original_coords if key not in kwargs}

        newdata_vars = {}
        for key in self.original_data_vars:
            if isinstance(self.original_data_vars[key], tuple):
                coords = [dim for dim in self.original_data_vars[key][0] if dim not in kwargs]
                newdata_vars[key] = (coords, self._variables[key].sel(**kwargs))
            else:
                newdata_vars[key] = self.original_data_vars[key].sel(**kwargs)
                
        return FastDataset(newdata_vars, newcoords, self.attrs)

    def __getitem__(self, name):
        return self._variables[name]
    
    def __getattr__(self, name):
        if name in self._variables:
            return self._variables[name]
        elif name in self.attrs:
            return self.attrs[name]
        assert False, "%s not a known item." % name
            

class FastDataArray(xr.DataArray):
    def __init__(self, data, coords, parentds):
        # Do not call __init__ on DataArray, to avoid time cost
        self.original_coords = coords
        self._values = self._data = data
        self.parentds = parentds

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

            return FastDataArray(values, self.original_coords, self.parentds)
        
        return func

    def reduce(self, func, dim=None, axis=None, keep_attrs=False, **kwargs):
        data = func(self._values)
        # Only handles no reduction or total reduction
        newshape = getattr(data, 'shape', ())
        if newshape == ():
            return FastDataArray(data, (), self.parentds)
        elif newshape == self.shape:
            return FastDataArray(data, self.original_coords, self.parentds)
        else:
            if dim is not None:
                axis = self.original_coords.index(dim)
            newcoords = list(self.original_coords)
            del newcoords[axis]
            return FastDataArray(data, newcoords, self.parentds)

    def sel(self, **kwargs):
        newcoords = tuple([self.original_coords[ii] for ii in range(len(self.original_coords)) if self.original_coords[ii] not in kwargs])
        if newcoords == self.original_coords:
            return self

        indices = [slice(None)] * len(self.original_coords)
        for dim in kwargs:
            axis = self.original_coords.index(dim)
            indices[axis] = self.parentds[dim]._values == kwargs[dim]
            
        return FastDataArray(self._data[tuple(indices)], newcoords, self.parentds)
    
    def __array__(self):
        return np.asarray(self._values)

    def __getitem__(self, inds):
        return self._data[inds]

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
        subds = FastDataset(newvars, coords={'time': timevar}, attrs={'year': year, 'region': region})
            
        yield region, subds

def merge(dss):
    all_data_vars = {}
    all_coords = {}
    all_attrs = {}

    for ds in dss:
        if isinstance(ds, FastDataset):
            for key in ds.original_coords:
                if key in all_coords:
                    all_coords[key] = assert_index_equal(all_coords[key], ds.original_coords[key])
                else:
                    all_coords[key] = ds.original_coords[key]
            for key in ds.original_data_vars:
                if key in all_coords:
                    all_coords[key] = assert_index_equal(all_coords[key], ds.original_coords[key])
                    continue
                all_data_vars[key] = ds.original_data_vars[key]
        else:
            for key in ds._dims:
                if key in all_coords:
                    all_coords[key] = assert_index_equal(all_coords[key], ds._dims[key])
                else:
                    all_coords[key] = ds._dims[key]
            for key in ds._variables:
                if key in all_coords:
                    all_coords[key] = assert_index_equal(all_coords[key], ds._variables[key])
                    continue
                all_data_vars[key] = (ds._variables[key].dims, ds._variables[key])

        if ds.attrs is not None:
            for key in ds.attrs:
                all_attrs[key] = ds.attrs[key]

    return FastDataset(all_data_vars, all_coords, all_attrs)

def concat(objs, dim=None):
    data_vars = {}
    dimdata_vars = {}
    coords = {}
    dimcoords = {}
    attrs = {}
    
    for ds in objs:
        if isinstance(ds, FastDataset):
            for key in ds.original_coords:
                if key == dim or dim in ds.original_coords[key].dims:
                    if key not in dimcoords:
                        dimcoords[key] = [ds.original_coords[key]]
                    else:
                        dimcoords[key].append(ds.original_coords[key])
                else:
                    if key in coords:
                        coords[key] = assert_index_equal(coords[key], ds.original_coords[key])
                    else:
                        coords[key] = ds.original_coords[key]
            for key in ds.original_data_vars:
                if key == dim:
                    if isinstance(ds.original_data_vars[key], tuple):
                        dimcoords[key].append(ds.original_data_vars[key][1])
                    else:
                        dimcoords[key].append(ds.original_data_vars[key]._data)
                elif dim in ds._variables[key].original_coords:
                    if key not in dimdata_vars:
                        dimdata_vars[key] = [ds.original_data_vars[key]]
                    else:
                        dimdata_vars[key].append(ds.original_data_vars[key])
                else:
                    data_vars[key] = ds.original_data_vars[key]
        else:
            for key in ds._dims:
                if key == dim or dim in ds._dims[key].dims:
                    if key not in dimcoords:
                        dimcoords[key] = [ds._dims[key]._data]
                    else:
                        dimcoords[key].append(ds._dims[key]._data)
                else:
                    if key in coords:
                        coords[key] = assert_index_equal(coords[key], ds._dims[key])
                    else:
                        coords[key] = ds._dims[key]
            for key in ds._variables:
                if key == dim:
                    dimcoords[key].append(ds._variables[key]._data)
                elif dim in ds._variables[key].dims:
                    if key not in datadata_vars:
                        dimdata_vars[key] = [ds._variables[key]._data]
                    else:
                        dimdata_vars[key].append(ds._variables[key]._data)
                else:
                    data_vars[key] = ds._variables[key]

        if ds.attrs is not None:
            for key in ds.attrs:
                attrs[key] = ds.attrs[key]

    for key in dimdata_vars:
        data_vars[key] = (dimdata_coords[key], np.concatenate(dimdata_vars[key], dimdata_coords[key].index(dim)))
    for key in dimcoords:
        coords[key] = np.concatenate(dimcoords[key])
        
    return FastDataset(data_vars, coords, attrs)

def assert_index_equal(one, two):
    if np.array(one).dtype != np.array(two).dtype:
        if not isinstance(one, int) and np.array(one).dtype == np.int32 and one[0] == 1 and one[-1] == len(one):
            one = len(one)
        if not isinstance(two, int) and np.array(two).dtype == np.int32 and two[0] == 1 and two[-1] == len(two):
            two = len(two)
        
    if isinstance(one, int) and isinstance(two, int):
        assert two == one
        return one
    elif isinstance(one, int):
        assert len(two) == one
        return two
    elif isinstance(two, int):
        assert len(one) == two
        return one
    else:
        assert np.array_equal(two, one), "Not equal: %s <> %s" % (str(two), str(one))
        return one

FastDataArray.__array_priority__ = 80
xr.core.ops.inject_binary_ops(FastDataArray)
    
if __name__ == '__main__':
    ds = FastDataset({'x': ('time', np.ones(3))}, {'time': np.arange(3)})
    print ds
    print ds.x
    print ds.x * 3