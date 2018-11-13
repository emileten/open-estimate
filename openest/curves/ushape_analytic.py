def derivative_clipping(ccs, mintemp):
    derivcoeffs = np.array(ccs) * np.arange(1, len(ccs) + 1) # Construct the derivative
    roots = filter(np.isreal, np.roots(derivcoeffs[::-1])) # only consider real roots
    
    deriv2coeffs = derivcoeffs[1:] * np.arange(1, len(derivcoeffs)) # Second derivative
    direction = np.polyval(deriv2coeffs[::-1], roots)
    rootvals = np.polyval(([0] + ccs)[::-1], roots)

    alllimits = set([-np.inf, np.inf])
    levels = [] # min level for each span
    spans = [] # list of tuples with spans
    for ii in range(len(roots)):
        if direction[ii] < 0: # ignore if turning back up
            levels.append(rootvals[ii])
            if roots[ii] < mintemp:
                spans.append((-np.inf, roots[ii]))
                alllimits.update([-np.inf, roots[ii]])
            else:
                spans.append((roots[ii], np.inf))
                alllimits.update([roots[ii], np.inf])
            
    xxlimits = np.sort(list(alllimits))
    yy = []
    for ii in range(len(xxlimits) - 1):
        level = -np.inf
        for jj in range(len(spans)):
            if spans[jj][0] <= xxlimits[ii] and spans[jj][1] >= xxlimits[ii+1]:
                level = max(level, levels[jj])
        yy.append(level)

    return xxlimits, yy

def UShapedCurve(curve, ccs, mintemp):
    baselevel = curve(mintemp)
    xxlimits, levels = derivative_clipping(ccs, mintemp)
    return MaximumCurve(ClippedCurve(ShiftedCurve(curve, -baselevel)),
                        StepCurve(xxlimits, levels - baselevel))
