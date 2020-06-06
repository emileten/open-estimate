import numpy as np
from openest.curves.basic import UnivariateCurve


class UShapedCurve(UnivariateCurve):
    def __init__(self, curve, midtemp, gettas, ordered=False, fillxxs=[], fillyys=[], direction='boatpose'):
        # Ordered only used for unit testing
        super(UShapedCurve, self).__init__(curve.xx)
        self.curve = curve
        self.midtemp = midtemp
        self.gettas = gettas
        self.ordered = ordered
        self.fillxxs = fillxxs
        self.fillyys = fillyys
        assert direction in ['boatpose', 'downdog'], "Unknown direction for u-shaped clipping."
        self.direction = direction

    def __call__(self, xs):
        values = self.curve(xs)
        tas = np.array(self.gettas(xs))
        
        if len(self.fillxxs) > 0:
            tas_saved = tas
            tas = np.concatenate((tas, self.fillxxs))
            values = np.concatenate((values, self.fillyys))

        order = np.argsort(tas)
        orderedtas = tas[order]
        orderedvalues = values[order]

        if len(self.fillxxs) > 0:
            tokeeps = order < len(tas_saved)
            lowkeeps = tokeeps[orderedtas < self.midtemp]
            highkeeps = tokeeps[orderedtas >= self.midtemp]

        lowvalues = orderedvalues[orderedtas < self.midtemp]
        if self.direction == 'boatpose':
            lowvalues2 = np.maximum.accumulate(lowvalues[::-1])
        elif self.direction == 'downdog':
            lowvalues2 = np.minimum.accumulate(lowvalues[::-1])

        highvalues = orderedvalues[orderedtas >= self.midtemp]
        if self.direction == 'boatpose':
            highvalues2 = np.maximum.accumulate(highvalues)
        elif self.direction == 'downdog':
            highvalues2 = np.minimum.accumulate(highvalues)

        if len(self.fillxxs) > 0:
            # Remove the fillins
            lowvalues2 = lowvalues2[lowkeeps[::-1]]
            highvalues2 = highvalues2[highkeeps]
        
        if self.ordered:
            return np.concatenate((lowvalues2[::-1], highvalues2))
        else:
            return np.concatenate((lowvalues2, highvalues2))

        
class UShapedDynamicCurve(UnivariateCurve):
    def __init__(self, curve, midtemp, gettas, unicurve, ordered=False, numfills=50, direction='boatpose'):
        super(UShapedDynamicCurve, self).__init__(curve.xx)
        self.curve = curve
        self.midtemp = midtemp
        self.gettas = gettas
        self.unicurve = unicurve
        self.ordered = ordered
        self.numfills = numfills
        self.direction = direction

    def __call__(self, xs):
        tas = self.gettas(xs)
        fillxxs = np.arange(np.min(tas), np.max(tas), self.numfills)[1:-1]
        fillyys = self.unicurve(tas)
        ucurve = UShapedCurve(self.curve, self.midtemp, self.gettas, self.ordered, fillxxs, fillyys, direction=self.direction)
        return ucurve(xs)

    
# Return tmarginal evaluated at the innermost edge of plateaus
class UShapedClipping(UnivariateCurve):
    def __init__(self, curve, tmarginal_curve, midtemp, gettas, ordered=False):
        super(UShapedClipping, self).__init__(curve.xx)
        self.curve = curve
        self.tmarginal_curve = tmarginal_curve
        self.midtemp = midtemp
        self.gettas = gettas
        self.ordered = ordered

    def __call__(self, xs):
        increasingvalues = self.curve(xs) # these are ordered as low..., high...
        increasingplateaus = np.diff(increasingvalues) == 0

        tas = self.gettas(xs)
        order = np.argsort(tas)
        orderedtas = tas[order]

        n_below = sum(orderedtas < self.midtemp)
        
        lowindicesofordered = np.arange(n_below)[::-1] # [N-1 ... 0]
        if len(lowindicesofordered) > 1:
            lowindicesofordered[np.concatenate(([False], increasingplateaus[:len(lowindicesofordered)-1]))] = n_below
            lowindicesofordered = np.minimum.accumulate(lowindicesofordered)
        
        highindicesofordered = np.arange(sum(orderedtas >= self.midtemp)) + n_below # [N ... T-1]
        if len(highindicesofordered) > 1:
            highindicesofordered[np.concatenate(([False], increasingplateaus[-len(highindicesofordered)+1:]))] = n_below
            highindicesofordered = np.maximum.accumulate(highindicesofordered)

        if len(xs.shape) == 2:
            increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered], :]), self.tmarginal_curve(xs[order[highindicesofordered], :]))) # ordered low..., high...
        else:
            increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered]]), self.tmarginal_curve(xs[order[highindicesofordered]]))) # ordered low..., high...
        increasingresults[increasingvalues <= 0] = 0 # replace truly clipped with 0

        if self.ordered:
            return np.concatenate((increasingresults[tas < self.midtemp][::-1], increasingresults[tas >= self.midtemp]))
        else:
            return increasingresults
