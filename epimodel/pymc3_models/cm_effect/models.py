import logging
import math

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T
from pymc3 import Model

from ..utils import geom_convolution, array_stats

log = logging.getLogger(__name__)


class BaseCMModel(Model):
    def __init__(self, data, model=None, name=""):
        super().__init__(name, model)
        self.d = data
        self.plot_trace_vars = set()
        self.trace = None

    def LogNorm(self, name, mean, log_var, plot_trace=True, hyperprior=None, **kwargs):
        """Create a lognorm variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        if hyperprior:
            # TODO
            pass

        v = pm.Lognormal(self.prefix + name, T.log(mean), log_var, **kwargs)
        # self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(self.prefix + name)

        return v

    def Det(self, name, exp, plot_trace=True):
        """Create a deterministic variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        v = pm.Deterministic(self.prefix + name, exp)
        # self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(self.prefix + name)
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

    def plot_CMReduction(self):
        assert self.trace is not None
        return pm.forestplot(
            self.trace, var_names=[self.prefix + "CMReduction"], credible_interval=0.9
        )

    def print_CMReduction(self):
        varname = self.prefix + "CMReduction"
        for i, c in enumerate(self.d.CMs):
            print(f"{i:2} {c:30} {varname:20} {array_stats(self.trace[varname][:,i])}")

    def print_var_per_country(self, varname):
        varname = self.prefix + varname
        for i, c in enumerate(self.d.Rs):
            print(
                f"{i:2} {self.d.rds[c].DisplayName:30} {varname:20} "
                f"{array_stats(self.trace[varname][i, ...])}"
            )

    def run(self, N, chains=2, cores=2):
        print(self.check_test_point())
        with self:
            self.trace = pm.sample(1000, chains=chains, cores=cores, init="adapt_diag")


class CMModelV2(BaseCMModel):
    """
    CM effect model V2 (lognormal prior)
    """

    def __init__(self, data, delay_mean=7.0):
        super().__init__(data)

        # Poisson distribution
        self.CMDelayProb = np.array(
            [
                delay_mean ** k * np.exp(-delay_mean) / math.factorial(k)
                for k in range(100)
            ]
        )
        assert abs(sum(self.CMDelayProb) - 1.0) < 1e-3

        # Shorten the distribution to have >99% of the mass
        self.CMDelayProb = self.CMDelayProb[np.cumsum(self.CMDelayProb) <= 0.999]
        # Cut off first days to have 90% of pre-existing intervention effect
        self.CMDelayCut = sum(np.cumsum(self.CMDelayProb) < 0.9)
        print(
            f"CM delay: mean {np.sum(self.CMDelayProb * np.arange(len(self.CMDelayProb)))}, len {len(self.CMDelayProb)}, cut at {self.CMDelayCut}"
        )

    def build_reduction_var(self, scale=0.1):
        """
        Less informative prior for CM reduction, allows values >1.0
        """
        # [CM] How much countermeasures reduce growth rate
        return self.LogNorm("CMReduction", 1.0, scale, shape=(self.nCMs,))

    def build(self):
        """
        Build the model variables.
        """
        CMReduction = self.build_reduction_var()

        # Window of active countermeasures extended into the past
        Earlier_ActiveCMs = self.d.get_ActiveCMs(
            self.d.Ds[0] - pd.DateOffset(self.CMDelayCut), self.d.Ds[-1]
        )

        # [region, CM, day] Reduction factor for each CM,C,D
        ActiveCMReduction = (
            T.reshape(CMReduction, (1, self.nCMs, 1)) ** Earlier_ActiveCMs
        )

        # [region, day] Reduction factor from CMs for each C,D (noise added below)
        GrowthReduction = self.Det(
            "GrowthReduction", T.prod(ActiveCMReduction, axis=1), plot_trace=False
        )

        # [region, day] Convolution of GrowthReduction by DelayProb along days
        DelayedGrowthReduction = self.Det(
            "DelayedGrowthReduction",
            geom_convolution(GrowthReduction, self.CMDelayProb, axis=1)[
                :, self.CMDelayCut :
            ],
            plot_trace=False,
        )

        # [] Baseline growth rate (wide prior OK, mean estimates ~10% daily growth)
        BaseGrowthRate = self.LogNorm("BaseGrowthRate", 1.2, 2.3)

        # [region] Region growth rate
        # TODO: Estimate growth rate variance
        RegionGrowthRate = self.LogNorm(
            "RegionGrowthRate", BaseGrowthRate, 0.3, shape=(self.nRs,)
        )

        # [region] Region unreliability as common scale multiplier of its:
        # * measurements (measurement unreliability)
        # * expected growth noise
        # TODO: Estimate good prior (but can be weak?)
        RegionScaleMult = self.LogNorm("RegionScaleMult", 1.0, 1.0, shape=(self.nRs,))

        # [region, day] The ideal predicted daily growth
        PredictedGrowth = self.Det(
            "PredictedGrowth",
            T.reshape(RegionGrowthRate, (self.nRs, 1)) * DelayedGrowthReduction,
            plot_trace=False,
        )

        # [region, day] The actual (still hidden) growth rate each day
        # TODO: Estimate noise varince (should be small, measurement variance below)
        #       Miscalibration: too low: time effects pushed into CMs, too high: explains away CMs
        RealGrowth = self.LogNorm(
            "RealGrowth",
            PredictedGrowth,
            RegionScaleMult.reshape((self.nRs, 1)) * 0.1,
            shape=(self.nRs, self.nDs),
            plot_trace=False,
        )

        # [region, day] Multiplicative noise applied to predicted growth rate
        RealGrowthNoise = self.Det(
            "RealGrowthNoise", RealGrowth / PredictedGrowth, plot_trace=False,
        )

        # [region] Initial size of epidemic (the day before the start, only those detected; wide prior OK)
        InitialSize = self.LogNorm("InitialSize", 1.0, 10, shape=(self.nRs,))

        # [region, day] The number of cases that would be detected with noiseless testing
        # (Noise source includes both false-P/N rates and local variance in test volume and targetting)
        # (Since we ony care about growth rates and assume consistent testing, it is fine to ignore real size)
        Size = self.Det(
            "Size",
            T.reshape(InitialSize, (self.nRs, 1)) * self.RealGrowth.cumprod(axis=1),
            plot_trace=False,
        )

        # [region, day] Cummulative tested positives
        Observed = self.LogNorm(
            "Observed",
            Size,
            0.4,  # self.RegionScaleMult.reshape((self.nRs, 1)) * 0.4,
            shape=(self.nRs, self.nDs),
            observed=self.d.Confirmed,
            plot_trace=False,
        )

        # [region, day] Multiplicative noise applied to predicted growth rate
        # Note: computed backwards, since self.Observed needs to be a distribution
        ObservedNoise = self.Det("ObservedNoise", Observed / Size, plot_trace=False,)


class CMModelV2g(CMModelV2):
    """
    CM effect model V2g (exp(-gamma) prior)
    """

    def build_reduction_var(self, alpha=0.5, beta=1.0):
        """
        CM reduction prior from ICL paper, only values <=1.0
        """
        # [CM] How much countermeasures reduce growth rate
        CMReductionGamma = pm.Gamma("CMReductionGamma", alpha, beta, shape=(self.nCMs,))
        return self.Det("CMReduction", T.exp((-1.0) * CMReductionGamma))
