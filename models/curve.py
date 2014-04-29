from univariate_model import UnivariateModel

class UnivariateCurve(UnivariateModel):
    def __init__(self, xx):
        super(UnivariateModel, self).__init__(xx_is_categorical=False, xx, scaled=True)
    
    def __call__(self, x):
        raise NotImplementedError("call not implemented")

    def get_xx(self):
        return self.xx
    
    def eval_pval(self, x, p, threshold=1e-3):
        return self(x)

class CurveCurve(UnivariateCurve):
    def __init__(self, xx, curve):
        super(UnivariateCurve, self).__init__(xx)
        self.curve = curve
    
    def __call__(self, x):
        return self.curve(x)

class AdaptableCurve(UnivariateCurve):
    def __init__(self, xx):
        super(AdaptableCurve, self).__init__(xx)

    def setup(self, yyyyddd, temps, **kw):
        pass

    def update():
        pass

