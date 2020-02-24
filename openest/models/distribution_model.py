from .ddp_model import DDPModel
from .spline_model import SplineModel, SplineModelConditional

class DistributionModel(DDPModel):
    # For distribution to be non-ddp, nice continuous form but requires continuous x
    def apply_as_distribution(self, model):
        if model.kind() == 'ddp_model':
            if len(model.xx) != len(self.yy):
                raise ValueError("Wrong number of values.")

            for ii in range(len(model.xx)):
                if model.get_xx()[ii] != self.get_yy()[ii]:
                    raise ValueError("Wrong values in distribution.")

            vv = self.lin_p().dot(model.lin_p())

            return DDPModel('ddp1', 'distribution', self.xx_is_categorical, self.get_xx(), model.yy_is_categorical, model.get_yy(), vv, scaled=self.scaled)
            
        elif model.kind() == 'spline_model':
            if len(model.xx) != len(self.yy):
                raise ValueError("Wrong number of values.")

            for ii in range(len(model.xx)):
                if model.get_xx()[ii] not in self.get_yy():
                    raise ValueError("Wrong values in distribution.")

            pp = self.lin_p()
            conditionals = []

            for ii in range(len(self.xx)):
                conds = []

                for jj in range(len(model.xx)):
                    original = model.get_conditional(model.get_xx()[jj])
                    conditional = SplineModelConditional(original.y0s, original.y1s, original.coeffs)
                    conditional.scale(pp[ii, self.get_yy().index(model.get_xx()[jj])])
                    conds.append(conditional)

                conditionals.append(SplineModelConditional.approximate_sum(conds))

            return SplineModel(self.xx_is_categorical, self.get_xx(), conditionals, scaled=self.scaled)
            
        else:
            raise ValueError("Unknown model type in apply_as_distribution")
