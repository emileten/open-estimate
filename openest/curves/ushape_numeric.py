import numpy as np
from openest.curves.basic import UnivariateCurve

class UShapedCurve(UnivariateCurve):
    """A curve that returns values that clips another curve to maintain a u-shape.

    Parameters
    ----------
    curve : UnivariateCurve
        The curve to be clipped.
    mintemp : float
        The temperature forced to be at the lowest point in the curve.
    gettas : function(xs) -> array_like
        Returns the core temperature variable from whatever the curve is called with.
    ordered : boolean or 'maintain'
        If 'maintain', return order according to tas; if True, increasing tas order
    fillxxs : array_like
        Temperature values for a grid of additional sampled temperatures
    fillyys : array_like
        The dependent variables that correspond to fillxxs
    """
    def __init__(self, curve, mintemp, gettas, ordered='maintain', fillxxs=[], fillyys=[]):
        super(UShapedCurve, self).__init__(curve.xx)
        self.curve = curve
        self.mintemp = mintemp
        self.gettas = gettas
        self.ordered = ordered
        self.fillxxs = fillxxs
        self.fillyys = fillyys

    def __call__(self, xs):
        values = self.curve(xs)
        tas = self.gettas(xs)

        # Add in the grid for completeness
        if len(self.fillxxs) > 0:
            tas_saved = tas
            tas = np.concatenate((tas, self.fillxxs))
            values = np.concatenate((values, self.fillyys))

        # Order into increasing values left and right of mintemp
        order = np.argsort(tas)
        orderedtas = tas[order]
        orderedvalues = values[order]

        lowvalues = orderedvalues[orderedtas < self.mintemp]
        lowvalues2 = np.maximum.accumulate(lowvalues[::-1])

        highvalues = orderedvalues[orderedtas >= self.mintemp]
        highvalues2 = np.maximum.accumulate(highvalues)

        if self.ordered == 'maintain':
            # Put it all back together
            increasing = np.concatenate((lowvalues2[::-1], highvalues2))
            # Undo the ordering
            tasorder = np.empty(len(increasing))
            tasorder[order] = increasing
            # Return just the given values
            return tasorder[:len(xs)]
        else:
            if len(self.fillxxs) > 0:
                tokeeps = order < len(tas_saved)
                lowkeeps = tokeeps[orderedtas < self.mintemp]
                highkeeps = tokeeps[orderedtas >= self.mintemp]

                # Remove the fillins
                lowvalues2 = lowvalues2[lowkeeps[::-1]]
                highvalues2 = highvalues2[highkeeps]

            if self.ordered:
                return np.concatenate((lowvalues2[::-1], highvalues2))
            else:
                return np.concatenate((lowvalues2, highvalues2))

# Return tmarginal evaluated at the innermost edge of plateaus
class UShapedClipping(UnivariateCurve):
    """Clip marginal effects to zero where a curve is u-clipped

    Parameters
    ----------
    curve : UnivariateCurve
        Curve that has been u-clipped (but may not be a UShapedCurve).
    tmarginal_curve : UnivariateCurve
        Curve representing the marginal effects wrt temperature.
    mintemp : float
        The temperature forced to be at the lowest point in the curve.
    gettas : function(xs) -> array_like
        Returns the core temperature variable from whatever the curve is called with.
    ordered : boolean or 'maintain'
        Must use the same value as the corresponding UShapedCurve
    """
    def __init__(self, curve, tmarginal_curve, mintemp, gettas, ordered='maintain'):
        super(UShapedClipping, self).__init__(curve.xx)
        self.curve = curve
        self.tmarginal_curve = tmarginal_curve
        self.mintemp = mintemp
        self.gettas = gettas
        self.ordered = ordered

    def __call__(self, xs):
        if self.ordered == 'maintain':
            tas = self.gettas(xs)
            order = np.argsort(tas)
            orderedtas = tas[order]

            # Order the clipped values to low..., high...
            clippedvalues = self.curve(xs)
            orderedvalues = clippedvalues[order]

            n_below = sum(orderedtas < self.mintemp)

            increasingvalues = np.concatenate((orderedvalues[:n_below][::-1], orderedvalues[n_below:]))

            # Find the plateaus
            increasingplateaus = np.diff(increasingvalues) == 0

            # Apply tmarginal at index just-before clipping (boat raises all)
            lowindicesofordered = np.arange(n_below)[::-1] # [N-1 ... 0]
            if len(lowindicesofordered) > 1:
                lowindicesofordered[np.concatenate(([False], increasingplateaus[:len(lowindicesofordered)-1]))] = n_below
                lowindicesofordered = np.minimum.accumulate(lowindicesofordered)
            
            highindicesofordered = np.arange(sum(orderedtas >= self.mintemp)) + n_below # [N ... T-1]
            if len(highindicesofordered) > 1:
                highindicesofordered[np.concatenate(([False], increasingplateaus[-len(highindicesofordered)+1:]))] = n_below
                highindicesofordered = np.maximum.accumulate(highindicesofordered)

            # Construct the results
            if len(xs.shape) == 2:
                increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered], :]), self.tmarginal_curve(xs[order[highindicesofordered], :]))) # ordered low..., high...
            else:
                increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered]]), self.tmarginal_curve(xs[order[highindicesofordered]]))) # ordered low..., high...
            increasingresults[increasingvalues <= 0] = 0 # replace truly clipped with 0

            orderedresults = np.concatenate((increasingresults[:n_below][::-1], increasingresults[n_below:]))
            
            # Undo the ordering
            tasorder = np.empty(len(orderedresults))
            tasorder[order] = orderedresults
            return tasorder
        else:
            increasingvalues = self.curve(xs) # these are ordered as low..., high...
            increasingplateaus = np.diff(increasingvalues) == 0

            tas = self.gettas(xs)
            order = np.argsort(tas)
            orderedtas = tas[order]
            
            n_below = sum(orderedtas < self.mintemp)
        
            lowindicesofordered = np.arange(n_below)[::-1] # [N-1 ... 0]
            if len(lowindicesofordered) > 1:
                lowindicesofordered[np.concatenate(([False], increasingplateaus[:len(lowindicesofordered)-1]))] = n_below
                lowindicesofordered = np.minimum.accumulate(lowindicesofordered)
            
            highindicesofordered = np.arange(sum(orderedtas >= self.mintemp)) + n_below # [N ... T-1]
            if len(highindicesofordered) > 1:
                highindicesofordered[np.concatenate(([False], increasingplateaus[-len(highindicesofordered)+1:]))] = n_below
                highindicesofordered = np.maximum.accumulate(highindicesofordered)

            if len(xs.shape) == 2:
                increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered], :]), self.tmarginal_curve(xs[order[highindicesofordered], :]))) # ordered low..., high...
            else:
                increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered]]), self.tmarginal_curve(xs[order[highindicesofordered]]))) # ordered low..., high...
            increasingresults[increasingvalues <= 0] = 0 # replace truly clipped with 0

            if self.ordered:
                return np.concatenate((increasingresults[tas < self.mintemp][::-1], increasingresults[tas >= self.mintemp]))
            else:
                return increasingresults
