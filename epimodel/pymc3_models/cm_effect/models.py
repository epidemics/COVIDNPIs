import logging

import numpy as np
import pymc3 as pm
import theano.tensor as T
from pymc3 import Model

from ..utils import geom_convolution

log = logging.getLogger(__name__)


class BaseCMModel(Model):
    def __init__(self, data, model=None, name=""):
        super().__init__(name, model)
        self.d = data
        self.prefix = '' if not(name) else (name + '_')
        # TODO: Use prefix
        self.plot_trace_vars = set()
        self.trace = None

    def LN(self, name, mean, log_var, plot_trace=True, hyperprior=None, shape=None):
        """Create a lognorm variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        if hyperprior:
            # TODO
            pass
        kws = {}
        if shape is not None:
            kws['shape'] = shape
        v = pm.Lognormal(name, T.log(mean), log_var, **kws)
        self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(name)
        return v

    def Det(self, name, exp, plot_trace=True):
        """Create a deterministic variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        v = pm.Deterministic(name, exp)
        self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(name)
        return v

    @property
    def nRs(self):
        return len(self.d.Rs)

    @property
    def nDs(self):
        return len(self.d.Ds)

    @property
    def nCMs(self):
        return len(self.d.CMs)

    def plot_traces(self):
        assert self.trace is not None
        return pm.traceplot(self.trace, var_names=list(self.plot_trace_vars))

    def plot_effect(self):
        assert self.trace is not None
        return pm.forestplot(
            self.trace, var_names=[f"CMReduction"], credible_interval=0.9
        )

    def run(self, N, chains=2, cores=2):
        print(self.check_test_point())
        with self:
            self.trace = pm.sample(1000, chains=chains, cores=cores, init="adapt_diag")


class CMModelV2(BaseCMModel):
    def __init__(self, data):
        super().__init__(data)

    def build_reduction_lognorm(self, scale=0.1):
        """Less informative prior for CM reduction, allows values >1.0"""
        # [CM] How much countermeasures reduce growth rate
        self.LN("CMReduction", 1.0, scale, shape=(self.nCMs,))

    def build_reduction_exp_gamma(self, alpha=0.5, beta=1.0):
        """CM reduction prior from ICL paper, only values <=1.0"""
        # [CM] How much countermeasures reduce growth rate
        self.CMReductionGamma = pm.Gamma(
            "CMReductionGamma", alpha, beta, shape=(self.nCMs,)
        )
        self.Det("CMReduction", T.exp((-1.0) * self.CMReductionGamma))

    def build_rates(self):
        """Build the part of model that predicts the growth rates."""

        # [] Baseline growth rate (wide prior OK, mean estimates ~10% daily growth)
        self.LN("BaseGrowthRate", 1.2, 2.3)

        # [region] Region growth rate
        # TODO: Estimate growth rate variance
        self.LN("RegionGrowthRate", self.BaseGrowthRate, 0.3, shape=(self.nRs,))

        # [region] Region unreliability as common scale multiplier of its:
        # * measurements (measurement unreliability)
        # * expected growth noise
        # TODO: Estimate good prior (but can be weak?)
        self.LN("RegionScaleMult", 1.0, 2.3, shape=(self.nRs,))

