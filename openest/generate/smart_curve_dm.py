import smart_curve

STOP # This is not used; instead using curvegen.get_terms

class DeltaMethodCurve(smart_curve.SmartCurve):
    def __init__(self, child, coefflen):
        super(DeltaMethodCurve, self).__init__()
        self.child = child
        self.coefflen = coefflen
    
    def __call__(self, ds):
        raise NotImplementedError("call not implemented")

    def format(self, lang):
        return self.child.format(lang)

class DeltaMethodCoefficientsCurve(DeltaMethodCurve):
    def __init__(self, coefflen, coeffs2d, variables, locations):
        super(DeltaMethodCurve, self).__init__(smart_curve.CoefficientsCurve(np.sum(coeffs2d, 1), variables), coefflen)
        self.coeffs2d = coeffs2d # NxM, for M pre-multiplied coefficients that interact with N variables
        self.variables = variables # N entries, strings
        self.locations = locations # NxM, 0 ... coefflen-1

    def __call__(self, ds):
        result = np.zeros((ds[self.variables[0]].shape[0], self.coefflen))
        for ii in range(len(self.variables)):
            #result += self.coeffs[ii] * ds[self.variables[ii]].values # TOO SLOW
            values = ds._variables[self.variables[ii]]._data.dot(self.coeffs2d[ii, :]) # Tx1 * 1xM
            result[self.locations[ii, :], :] = values
            
        return result

class DeltaMethodZeroInterceptPolynomialCurve(DeltaMethodCoefficientsCurve):
    def __init__(self, coefflen, coeffs2d, variables, locations, allow_raising=False, descriptions={}):
        super(DeltaMethodZeroInterceptPolynomialCurve, self).__init__(coefflen, coeffs2d, variables, locations)
        assert not allow_raising, "Raising not supported yet for delta-method calculations."
        self.child = smart_curve.ZeroInterceptPolynomialCurve(np.sum(coeffs2d, 1), variables, False, descriptions)
