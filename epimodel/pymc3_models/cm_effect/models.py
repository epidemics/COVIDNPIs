import logging

import numpy as np
import pymc3 as pm
import theano.tensor as T
from pymc3 import Model

from epimodel.pymc3_models.utils import geom_convolution

log = logging.getLogger(__name__)


class BaseCMModel(Model):
    def __init__(self, data, name="", model=None):
        super().__init__(name, model)
        self.d = data
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
            kws["shape"] = shape
        v = pm.Lognormal(name, mean, log_var, **kws)
        self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(name)
        return v

    def ObservedLN(self, name, mean, log_var, observed, plot_trace=True, shape=None):
        """Create a lognorm variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")

        kws = {}
        if shape is not None:
            kws["shape"] = shape

        v = pm.Lognormal(name, pm.math.log(mean), log_var, observed=observed, **kws)
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
            self.trace = pm.sample(N, chains=chains, cores=cores, init="adapt_diag")


class CMModelV2(BaseCMModel):
    def __init__(self, data, name="", model=None):
        super().__init__(data, name, model)

        self.CMDelayCut = 10
        self.DelayProb = np.array([0.00, 0.01, 0.02, 0.06, 0.10, 0.13, 0.15, 0.15, 0.13, 0.10, 0.07, 0.05, 0.03])
        self.DailyGrowthNoiseMultiplier = 0.1
        self.ConfirmedCasesNoiseMultiplier = 0.4

    def build_reduction_lognorm(self, scale=0.1):
        """Less informative prior for CM reduction, allows values >1.0"""
        # [CM] How much countermeasures reduce growth rate
        self.LN("CMReduction", 0, scale, shape=(self.nCMs,))

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
        self.LN("BaseGrowthRate", np.log(1.2), 2.0)

        # [region] Region growth rate
        # TODO: Estimate growth rate variance
        self.LN("RegionGrowthRate", pm.math.log(self.BaseGrowthRate), 0.3, shape=(self.nRs,))

        # [region] Region unreliability as common scale multiplier of its:
        # * measurements (measurement unreliability)
        # * expected growth noise
        # TODO: Estimate good prior (but can be weak?)
        self.LN("RegionScaleMult", 0.0, 1.0, shape=(self.nRs,))

        try:
            self.ActiveCMReduction = T.reshape(self.CMReduction, (1, self.nCMs, 1)) ** self.d.ActiveCMs
        except AttributeError:
            # TODO: make a custom exception class for not build error or similar
            raise Exception("Missing CM Reduction Prior; have you built that?")

        self.Det("GrowthReduction", T.prod(self.ActiveCMReduction, axis=1))
        self.DelayedGrowthReduction = geom_convolution(self.GrowthReduction, self.DelayProb, axis=1)[:, self.CMDelayCut:]
        self.Det("PredictedGrowth", T.reshape(self.RegionGrowthRate, (self.nRs, 1)) * self.DelayedGrowthReduction)
        self.LN("DailyGrowth", pm.math.log(self.PredictedGrowth),
                self.RegionScaleMult.reshape((self.nRs, 1)) * self.DailyGrowthNoiseMultiplier,
                shape=(self.nRs, self.nDs - self.CMDelayCut),
                )

    def build_output_model(self):
        self.LN("InitialSize", 0, 10, shape=(self.nRs,))
        try:
            self.Det("Size", T.reshape(self.InitialSize, (self.nRs, 1)) * self.DailyGrowth.cumprod(axis=1))
        except AttributeError:
            raise Exception("Missing CM Reduction Prior; have you built that?")

        Observed = pm.Lognormal("Observed",
                                pm.math.log(self.Size), self.RegionScaleMult.reshape((self.nRs, 1)) * self.ConfirmedCasesNoiseMultiplier,
                                shape=(self.nRs, self.nDs - self.CMDelayCut),
                                observed=self.d.Confirmed[:, self.CMDelayCut:])
        # self.ObservedLN("Observed",
        #         self.Size,
        #         self.RegionScaleMult.reshape((self.nRs, 1)) * self.ConfirmedCasesNoiseMultiplier,
        #         shape=(self.nRs, self.nDs - self.CMDelayCut),
        #         observed=self.d.Confirmed[:, self.CMDelayCut:]
        #         )

    def build_all(self):
        self.build_reduction_lognorm()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")
