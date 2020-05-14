# class LinearExtrapolationCurve(UnivariateCurve):
#     def __init__(self, curve, limits, margin, scaling, getindep):
#         super(UShapedCurve, self).__init__(curve.xx)
#         self.curve = curve
#         self.limits = limits
#         self.margin = margin
#         self.scaling = scaling
#         self.getindep = getindep

#     def __call__(self, xs):
#         values = self.curve(xs)
#         indeps = self.getindep(xs)

#         known_affines = {}
#         for ii, indep, edge, invector in self.beyond(indeps):
#             if edge not in known_affines:
#                 y0 = self.curve(indep + invector)
#                 y1 = self.curve(indep + invector + self.margin * invector / numpy.linalg.norm(invector))
#                 slope = self.scaling * (y1 - y0) / self.margin
#                 known_affines[edge] = (y0, slope)

#             depen = known_affines[edge][0] + sum(known_affines[edge][0] * -invector)
#             values[ii] = depen

import numpy as np
