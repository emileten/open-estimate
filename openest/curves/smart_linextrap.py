import numpy as np
from . import linextrap
from openest.generate.smart_curve import SmartCurve

class LinearExtrapolationCurve(SmartCurve):
    """Linear extrapolation clipping curve which takes a xarray Dataset.

    Parameters
    ----------
    curve : SmartCurve
        Curve to be clipped
    indepvars : sequence
        The subset of variables that are considered independent
    bounds : dict or list
        Either a dictionary of {var: (lower, upper)} bounds or a polytope
    margins : dict
        A dict with margin for each variable
    scaling : float
        A factor that the slope is scaled by
    """
    def __init__(self, curve, indepvars, bounds, margins, scaling):
        super(LinearExtrapolationCurve, self).__init__()
        assert isinstance(bounds, list) or isinstance(bounds, dict)
        
        self.curve = curve
        self.indepvars = indepvars # need this order in case of polytope bounds
        self.bounds = {kk: bounds[indepvars[kk]] for kk in range(len(indepvars))} # convert into indexed bounds
        self.margins = [margins[indepvars[kk]] for kk in range(len(indepvars))] # convert into ordered list
        self.scaling = scaling

    def __call__(self, ds):
        """Returns the projected variables after clipping.

        See linextrap.LinearExtrapolationCurve for details.

        Parameters
        ----------
        ds : xarray.Dataset
            Weather variables including indepvars.

        Returns
        -------
        array_like
            A sequence of values after clipping.
        """
        values = self.curve(ds)
        indeps = np.stack((ds[indepvar] for indepvar in self.indepvars), axis=-1)

        return linextrap.replace_oob(values, indeps, self.curve.get_univariate(), self.bounds, self.margins, self.scaling)

    def format(self, lang):
        raise NotImplementedError()
        # elements = SmartCurve.format_call(lang, self.curve, self.indepvars)
        # if lang == 'latex':
        #     elements['main'] = FormatElement(r"\begin{cases} %s & \text{if within bounds} \\ \text{linear extrapolation} & \text{otherwise} \end{cases}", elements['main'].dependencies)
            
