# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, Distributed Meta-Analysis System
################################################################################


__copyright__ = "Copyright 2014, Distributed Meta-Analysis System"

__author__ = "James Rising"
__credits__ = ["James Rising", "Solomon Hsiang"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import copy

class Model(object):
    """
    Model class
    
    Top level Model class, from which all specific model derive.  All
    models should implement most of these functions (with the notable
    exceptions of merge and combine).
    """
    mergers = {}
    combiners = {}

    def __init__(self, scaled=True):
        self.scaled = scaled

    def kind(self):
        return 'unknown'

    def copy(self):
        return copy.deepcopy(self)

    # Determine the conditional probability density at ys
    def to_points_at(self, x, ys):
        """Conditional Probability Density Evaluation

        Returns unscaled probability density values for given values of $x$
        and $y$: $f(y | x)$."""
        raise NotImplementedError("to_points_at not implemented")
    
    def eval_pval(self, x, p, threshold=1e-3):
        """Inverse CDF Evaluation

        Returns the value of $y$ that corresponds to a given p-value:
        $F^{-1}(p | x)$."""
        raise NotImplementedError("eval_pval not implemented")

    def scale_y(self, a):
        """Rescaling of the Parameter Dimension

        Produces a new conditional PDF with the $y$ dimension scaled by a
        constant: $p(z | x) = p(\frac{y}{a} | x)$."""
        raise NotImplementedError("scale_y not implemented")

    def scale_p(self, a):
        """Raise the distribution to the power 'a' and rescales.

        Returns:
          self: modifies this model and returns it
        """
        raise NotImplementedError("scale_p not implemented")

    def get_mean(self, x=None):
        """E[Y | X]"""
        assert self.scaled, "Cannot take mean of unscaled distribution."
        
        return np.nan

    def get_sdev(self, x=None):
        """sqrt Var[Y | X]"""
        assert self.scaled, "Cannot take mean of unscaled distribution."
        
        return np.nan

    def draw_sample(self, x=None):
        """Produce a sample value of y from the conditional distribution."""
        raise NotImplementedError("draw_sample not implemented")

    def attribute_list(self):
        return []

    def get_attribute(self, title):
        raise ValueError(title + " not available")

    @staticmethod
    def merge(models):
        """Pooling Merging

        Each form provides methods for producing a pooled parameter
        estimate from multiple parameter estimates.  These could all be
        parameter estimates with the same form, or with two different forms:
        $p_1(y | x) p_2(y | x)$.
        """
        if len(models) == 1:
            return models[0]
        
        # Sort into groups of models
        groups = {}
        for ii in range(len(models)):
            mic = models[ii].kind()
            if mic not in groups:
                groups[mic] = []
            groups[mic].append(models[ii])

        merged = []
        for mic in groups:
            merged.append(Model.mergers[mic](groups[mic]))

        result = merged[0]
        for ii in range(1, len(merged)):
            try:
                result = Model.mergers[result.kind() + "+" + merged[ii].kind()]([result, merged[ii]])
            except Exception as ex:  # CATBELL
                import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
                result = Model.mergers[merged[ii].kind() + "+" + result.kind()]([merged[ii], result])

        return result

    @staticmethod
    def combine(models, factors):
        """Construct a weighted sum over the shared values of x

        Each form provides methods for constructing the distribution of the
        sum of multiple parameters, which is generally constructed by
        performing the convolution: $p(y + z | x) = p_y(y | x) * p_z(z | x)$.
        """
        sofar = None
        for ii in range(len(models)):
            model = models[ii]

            scaled = model.copy().scale_y(factors[ii])
            if sofar is None:
                sofar = scaled
            else:
                try:
                    sofar = Model.combiners[sofar.kind() + "+" + scaled.kind()](sofar, scaled)
                except Exception as ex:  # CATBELL
                    import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
                    sofar = Model.combiners[scaled.kind() + "+" + sofar.kind()](scaled, sofar)

        return sofar

class Attribute(object):
    """An attribute is an arbitrary piece of information about a
    model, available from the attribute functions on Model."""
    def __init__(self, title, description, reference, subtitle, value, comments, source):
        self.title = title
        self.description = description
        self.reference = reference
        self.subtitle = subtitle
        self.value = value
        self.comments = comments
        self.source = source
