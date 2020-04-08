import logging

import pymc3 as pm
from pymc3 import Model
import numpy as np
import theano.tensor as T

from epimodel.pymc3_models.utils import geom_convolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseCountermeasureModel(Model):

    def __init__(self, name=" ", model=None):
        super().__init__(name, model)
        self.trace = None

    def sample(self, N, chains, cores, init, plot_trace=True):
        # looks hacky, but this object is a model
        with self.model:
            self.trace = pm.sample(N, chains=chains, cores=cores, init=init)

        if plot_trace:
            _ = pm.traceplot(self.trace, var_names=self.get_trace_vars())

    def get_trace_vars(self):
        raise NotImplementedError()

    def plot_inferred_cm_effect(self):
        raise NotImplementedError()


class CountermeasureModelV2(BaseCountermeasureModel):

    def __init__(self, dataset_size, ActiveCMs, DelayProb, CMDelayCut, Confirmed, name=" ", model="None"):
        super().__init__(name, model)

        (nCs, nCMs, nDs) = dataset_size
        self.nCs = nCs
        self.nCMs = nCMs
        self.nDs = nDs

        BaseGrowthRate = pm.Lognormal("BaseGrowthRate", np.log(1.2), 2.0)

        # [country] Initial size of epidemic (the day before the start, only those detected; wide prior OK)
        InitialSize = pm.Lognormal("InitialSize", 0.0, 10, shape=(nCs,))

        # [country] Country growth rate
        # TODO: Estimate growth rate variance
        CountryGrowthRate = pm.Lognormal("CountryGrowthRate", pm.math.log(BaseGrowthRate), 0.3, shape=(nCs,))

        # [country] Country unreliability as common scale multiplier of its:
        # * measurements (measurement unreliability)
        # * expected growth noise
        # TODO: Estimate good prior (but can be weak?)
        CountryScaleMult = pm.Lognormal("CountryScaleMult", 0.0, 1.0, shape=(nCs,))  # Weak prior!

        # [CM] How much countermeasures reduce growth rate
        # TODO: Possibly use another distribution
        CMReduction = pm.Lognormal("CMReduction", 0.0, 0.1, shape=(nCMs,))

        # [country, CM, day] Reduction factor for each CM,C,D
        ActiveCMReduction = T.reshape(CMReduction, (1, nCMs, 1)) ** ActiveCMs

        # [country, day] Reduction factor from CMs for each C,D (noise added below)
        GrowthReduction = pm.Deterministic("GrowthReduction", T.prod(ActiveCMReduction, axis=1))

        # [country, day] Convolution of GrowthReduction by DelayProb along days
        DelayedGrowthReduction = geom_convolution(GrowthReduction, DelayProb, axis=1)

        # Erase early DlayedGrowthRates in first ~10 days (would assume them non-present otherwise!)
        DelayedGrowthReduction = DelayedGrowthReduction[:, CMDelayCut:]

        # [country, day - CMDelayCut] The ideal predicted daily growth
        PredictedGrowth = pm.Deterministic("PredictedGrowth",
                                           T.reshape(CountryGrowthRate, (nCs, 1)) * DelayedGrowthReduction)

        # [country, day - CMDelayCut] The actual (still hidden) growth rate each day
        # TODO: Estimate noise varince (should be small, measurement variance below)
        #       Miscalibration: too low: time effects pushed into CMs, too high: explains away CMs
        DailyGrowth = pm.Lognormal("DailyGrowth",
                                   pm.math.log(PredictedGrowth),
                                   CountryScaleMult.reshape((nCs, 1)) * 0.1, shape=(nCs, nDs - CMDelayCut))

        # Below I assume plain exponentia growth of confirmed rather than e.g. depending on the remaining
        # susceptible opulation etc.

        # AdditiveDailyGrowth = DailyGrowth
        # [country, day - CMDelayCut] The number of cases that would be detected with noiseless testing
        # (Noise source includes both false-P/N rates and local variance in test volume and targetting)
        # (Since we ony care about growth rates and assume consistent testing, it is fine to ignore real size)
        Size = pm.Deterministic("Size", T.reshape(InitialSize, (nCs, 1)) * DailyGrowth.cumprod(axis=1))

        # [country, day - CMDelayCut] Cummulative tested positives
        Observed = pm.Lognormal("Observed",
                                pm.math.log(Size), CountryScaleMult.reshape((nCs, 1)) * 0.4,
                                shape=(nCs, nDs - CMDelayCut),
                                observed=Confirmed[:, CMDelayCut:])

        logger.info(f"Checking Model Test Point\n{self.check_test_point()}\n")

        self.trace_var_names = [f"{self.name}_{var}" for var in
                                ["BaseGrowthRate", "CountryGrowthRate", "CMReduction", "CountryScaleMult"]]

    def get_trace_vars(self):
        return self.trace_var_names

    def plot_inferred_cm_effect(self):
        if self.trace is None:
            raise AssertionError("Chain has not been sampled from - please call model.sample()")
        # could be moved into the super
        pm.forestplot(self.trace, var_names=[f"{self.name}_CMReduction"], credible_interval=0.9)
