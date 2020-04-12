import copy
import logging
import os
from datetime import datetime

import numpy as np
import pymc3 as pm
import theano.tensor as T
from pymc3 import Model

from epimodel.pymc3_models.utils import geom_convolution

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt


class BaseCMModel(Model):
    def __init__(self, data, name="", model=None):
        super().__init__(name, model)
        self.d = data
        self.plot_trace_vars = set()
        self.trace = None
        self.heldout_day_labels = None

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

        v = pm.Lognormal(name, mean, log_var, observed=observed, **kws)
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
        with self.model:
            self.trace = pm.sample(N, chains=chains, cores=cores, init="adapt_diag")

    def heldout_days_validation_plot(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None
        assert self.heldout_day_labels is not None

        labels = self.heldout_day_labels
        predictions = self.trace["HeldoutDays"]

        means = np.mean(predictions, axis=0)
        li = np.percentile(predictions, 2.5, axis=0)
        ui = np.percentile(predictions, 97.5, axis=0)

        N_heldout_days = means.shape[1]

        plt.figure(figsize=(12, 4), dpi=300)

        max_val = 10 ** np.ceil(np.log10(max(np.max(ui), np.max(labels))))
        min_val = 10 ** np.floor(np.log10(min(np.min(li), np.min(labels))))

        for day in range(N_heldout_days):
            plt.subplot(1, N_heldout_days, day + 1)
            x = labels[:, day]
            err = np.array([means[:, day] - li[:, day], -means[:, day] + ui[:, day]])
            y = means[:, day]
            plt.errorbar(x, y, yerr=err, linestyle=None, fmt='ko')
            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.plot([0, 10 ** 6], [0, 10 ** 6], '-r')
            plt.xlim([min_val, max_val])
            plt.ylim([min_val, max_val])
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"Heldout Day {day + 1}")

        plt.tight_layout()
        if save_fig:
            datetime_str = datetime.now().strftime("%d-%m;%H:%M")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            log.info(f"Saving Day Validation Plot at {os.path.abspath(output_dir)}")
            plt.savefig(f"{output_dir}/HeldoutDaysValidation_{datetime_str}.pdf")


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
        self.LN("RegionScaleMult", 0.0, 1.0, shape=(self.nRs,), plot_trace=False)

        self.ActiveCMReduction = T.reshape(self.CMReduction, (1, self.nCMs, 1)) ** self.d.ActiveCMs
        self.Det("GrowthReduction", T.prod(self.ActiveCMReduction, axis=1), plot_trace=False)
        self.DelayedGrowthReduction = geom_convolution(self.GrowthReduction, self.DelayProb, axis=1)[:,
                                      self.CMDelayCut:]
        self.Det("PredictedGrowth", T.reshape(self.RegionGrowthRate, (self.nRs, 1)) * self.DelayedGrowthReduction,
                 plot_trace=False)
        self.LN("DailyGrowth", pm.math.log(self.PredictedGrowth),
                self.RegionScaleMult.reshape((self.nRs, 1)) * self.DailyGrowthNoiseMultiplier,
                shape=(self.nRs, self.nDs - self.CMDelayCut),
                plot_trace=False)

    def build_output_model(self):
        self.LN("InitialSize", 0, 10, shape=(self.nRs,))
        try:
            self.Det("Size", T.reshape(self.InitialSize, (self.nRs, 1)) * self.DailyGrowth.cumprod(axis=1),
                     plot_trace=False)
        except AttributeError:
            raise Exception("Missing CM Reduction Prior; have you built that?")

        self.ObservedLN("Observed",
                        pm.math.log(self.Size),
                        self.RegionScaleMult.reshape((self.nRs, 1)) * self.ConfirmedCasesNoiseMultiplier,
                        shape=(self.nRs, self.nDs - self.CMDelayCut),
                        observed=self.d.Confirmed[:, self.CMDelayCut:],
                        plot_trace=False
                        )

    def build_heldout_days_output_model(self, N_days_holdout):
        # effectively the same as the normal output model except we reduce the number of observations given
        self.LN("InitialSize", 0, 10, shape=(self.nRs,))
        try:
            self.Det("Size", T.reshape(self.InitialSize, (self.nRs, 1)) * self.DailyGrowth.cumprod(axis=1),
                     plot_trace=False)
        except AttributeError:
            raise Exception("Missing CM Reduction Prior; have you built that?")

        # now we only observe the days until the final days.
        self.ObservedLN("Observed",
                        pm.math.log(self.Size[:, :-N_days_holdout]),
                        self.RegionScaleMult.reshape((self.nRs, 1)) * self.ConfirmedCasesNoiseMultiplier,
                        shape=(self.nRs, self.nDs - self.CMDelayCut - N_days_holdout),
                        observed=self.d.Confirmed[:, self.CMDelayCut:-N_days_holdout],
                        plot_trace=False
                        )

        self.LN("HeldoutDays",
                pm.math.log(self.Size[:, -N_days_holdout:]),
                self.RegionScaleMult.reshape((self.nRs, 1)) * self.ConfirmedCasesNoiseMultiplier,
                shape=(self.nRs, N_days_holdout), plot_trace=True
                )

        self.heldout_day_labels = self.d.Confirmed[:, -N_days_holdout:]

    def build_heldout_region_heldout_priors(self, initial_size_dist, initial_size_kwargs, region_growth_dist,
                                            region_growth_kwargs, region_scale_dist, region_scale_kwargs):
        with self.model:
            # add these arbitary distributions. don't add them to the plots.
            self.HeldoutInitialSize = initial_size_dist(name="HeldoutInitialSize", **initial_size_kwargs)
            self.HeldoutBaseGrowthRate = region_growth_dist(name="HeldoutBaseGrowthRate", **region_growth_kwargs)
            self.HeldoutScale = region_scale_dist(name="HeldoutScale", **region_scale_kwargs)

    def compute_heldout_region_indices(self, heldout_region):
        observed_regions = copy.deepcopy(self.d.Rs)
        observed_regions_indx = list(range(self.nRs))
        heldout_region_indx = observed_regions.index(heldout_region)
        observed_regions.remove(heldout_region)
        observed_regions_indx.remove(heldout_region_indx)

        return heldout_region_indx, observed_regions, np.array(observed_regions_indx, dtype=np.int8)

    def build_heldout_region_growth_model(self, heldout_region):

        # we need to add all but the excluded country, run as usual.
        # need to provide initial estimates for the missing country
        # for this model class, need initial size and base growth rate input

        heldout_region_indx, observed_regions, observed_regions_indx = self.compute_heldout_region_indices(
            heldout_region)

        self.LN("BaseGrowthRate", np.log(1.2), 2.0)
        self.LN("RegionGrowthRate", pm.math.log(self.BaseGrowthRate), 0.3, shape=(self.nRs - 1,))
        self.LN("RegionScaleMult", 0.0, 1.0, shape=(self.nRs - 1,), plot_trace=False)

        self.ActiveCMReduction = T.reshape(self.CMReduction, (1, self.nCMs, 1)) ** self.d.ActiveCMs[
                                                                                   observed_regions_indx, :]
        self.Det("GrowthReduction", T.prod(self.ActiveCMReduction, axis=1), plot_trace=False)
        self.DelayedGrowthReduction = geom_convolution(self.GrowthReduction, self.DelayProb, axis=1)[:,
                                      self.CMDelayCut:]
        self.Det("PredictedGrowth", T.reshape(self.RegionGrowthRate, (self.nRs - 1, 1)) * self.DelayedGrowthReduction,
                 plot_trace=False)
        self.LN("DailyGrowth", pm.math.log(self.PredictedGrowth),
                self.RegionScaleMult.reshape((self.nRs - 1, 1)) * self.DailyGrowthNoiseMultiplier,
                shape=(self.nRs - 1, self.nDs - self.CMDelayCut),
                plot_trace=False)

        self.HeldoutCMReduction = T.reshape(self.CMReduction, (1, self.nCMs, 1)) ** self.d.ActiveCMs[heldout_region_indx, :]
        self.Det("HeldoutGrowthReduction", T.prod(self.HeldoutCMReduction, axis=1), plot_trace=False)
        self.DelayedHeldoutGrowthReduction = geom_convolution(self.HeldoutGrowthReduction, self.DelayProb, axis=1)[:,
                                      self.CMDelayCut:]
        self.Det("HeldoutPredictedGrowth",
                 T.reshape(self.HeldoutBaseGrowthRate, (1, 1)) * self.DelayedHeldoutGrowthReduction,
                 plot_trace=False)
        self.LN("HeldoutDailyGrowth", pm.math.log(self.HeldoutPredictedGrowth),
                self.HeldoutScale.reshape((1, 1)) * self.DailyGrowthNoiseMultiplier,
                shape=(1, self.nDs - self.CMDelayCut),
                plot_trace=False)

    def build_heldout_region_observation_model(self, heldout_region):
        heldout_region_indx, observed_regions, observed_regions_indx = self.compute_heldout_region_indices(
            heldout_region)

        self.LN("InitialSize", 0, 10, shape=(self.nRs - 1,))
        self.Det("Size", T.reshape(self.InitialSize, (self.nRs - 1, 1)) * self.DailyGrowth.cumprod(axis=1),
                 plot_trace=False)

        self.ObservedLN("Observed",
                        pm.math.log(self.Size),
                        self.RegionScaleMult.reshape((self.nRs - 1, 1)) * self.ConfirmedCasesNoiseMultiplier,
                        shape=(self.nRs, self.nDs - self.CMDelayCut),
                        observed=self.d.Confirmed[observed_regions_indx, self.CMDelayCut:],
                        plot_trace=False
                        )

        self.Det("HeldoutSize", T.reshape(self.HeldoutInitialSize, (1, 1)) * self.HeldoutDailyGrowth.cumprod(axis=1),
                 plot_trace=False)
        self.LN("HeldoutObserved",
                pm.math.log(self.HeldoutSize),
                self.HeldoutScale * self.ConfirmedCasesNoiseMultiplier,
                shape=(1, self.nDs - self.CMDelayCut),
                plot_trace=False
                )
        self.HeldoutConfirmed = self.d.Confirmed[heldout_region_indx, :]

    def build_all(self):
        self.build_reduction_lognorm()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")
