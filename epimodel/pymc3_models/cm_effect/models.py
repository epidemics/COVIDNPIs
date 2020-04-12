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
        assert self.HeldoutDays is not None

        for indx, ho_day in enumerate(self.HeldoutDays):
            labels = self.d.Confirmed[:, ho_day]
            predictions = self.trace["ObservedSize"][:, :, ho_day - self.CMDelayCut]

            means = np.mean(predictions, axis=0)
            li = np.percentile(predictions, 2.5, axis=0)
            ui = np.percentile(predictions, 97.5, axis=0)
            err = np.array([means - li, -means + ui])

            max_val = 10 ** np.ceil(np.log10(max(np.max(ui), np.max(labels))))
            min_val = 10 ** np.floor(np.log10(min(np.min(li), np.min(labels))))

            plt.figure(figsize=(4, 3), dpi=300)
            plt.errorbar(labels, means, yerr=err, linestyle=None, fmt="ko")
            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.plot([0, 10 ** 6], [0, 10 ** 6], "-r")
            plt.xlim([min_val, max_val])
            plt.ylim([min_val, max_val])
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"Heldout Day {ho_day + 1}")
            plt.tight_layout()
            if save_fig:
                datetime_str = datetime.now().strftime("%d-%m;%H:%M")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                log.info(f"Saving Day Validation Plot at {os.path.abspath(output_dir)}")
                plt.savefig(
                    f"{output_dir}/HeldoutDaysValidation_d{ho_day}_t{datetime_str}.pdf"
                )

    def heldout_regions_validation_plot(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None
        assert self.HeldoutRegions is not None

        for indx, region_indx in enumerate(self.HR_indxs):
            days = self.d.Ds
            days_x = np.arange(len(days))
            labels = self.d.Confirmed[region_indx, days_x]
            predictions = self.trace["HeldoutObserved"][:, indx]
            cut_days = self.CMDelayCut

            plt.figure(figsize=(4, 3), dpi=300)
            pred_y = np.mean(predictions, axis=0).flatten()
            li_y = np.percentile(predictions, 2.5, axis=0).T.flatten()
            ui_y = np.percentile(predictions, 97.5, axis=0).T.flatten()
            yerr = np.array([pred_y - li_y, ui_y - pred_y])

            max_val = 10 ** np.ceil(np.log10(max(np.max(ui_y), np.max(labels))))
            min_val = 10 ** np.floor(np.log10(min(np.min(li_y), np.min(labels))))

            plt.errorbar(
                days_x[cut_days:], np.mean(predictions, axis=0).T, yerr=yerr, zorder=-1
            )
            plt.plot(days_x[cut_days:], labels[cut_days:], "-x", MarkerSize=3, zorder=0)
            ax = plt.gca()
            ax.set_yscale("log")
            plt.ylim([min_val, max_val])
            locs = np.arange(cut_days, len(days), 5)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("Confirmed Cases")
            plt.title(f"Heldout Region: {self.HeldoutRegions[indx]}")
            plt.xlabel("Date")

            if save_fig:
                datetime_str = datetime.now().strftime("%d-%m;%H:%M")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                log.info(f"Saving Day Validation Plot at {os.path.abspath(output_dir)}")
                plt.savefig(
                    f"{output_dir}/HeldoutRegionValidation_r{self.HeldoutRegions[indx]}_t{datetime_str}.pdf"
                )


class CMModelFlexibleV2(BaseCMModel):
    def __init__(
        self, data, heldout_days=None, heldout_regions=None, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.CMDelayCut = 10
        self.DelayProb = np.array(
            [
                0.00,
                0.01,
                0.02,
                0.06,
                0.10,
                0.13,
                0.15,
                0.15,
                0.13,
                0.10,
                0.07,
                0.05,
                0.03,
            ]
        )
        self.DailyGrowthNoiseMultiplier = 0.1
        self.ConfirmedCasesNoiseMultiplier = 0.4

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))

        if heldout_regions is not None:
            self.HeldoutRegions = copy.deepcopy(heldout_regions)
        else:
            self.HeldoutRegions = []

        if heldout_days is not None:
            self.HeldoutDays = copy.deepcopy(heldout_days)
            self.ObservedDaysIndx = np.delete(
                self.ObservedDaysIndx, np.array(self.HeldoutDays) - self.CMDelayCut
            )
        else:
            self.HeldoutDays = []

        # useful things for heldout stuff
        self.nORs = self.nRs - len(self.HeldoutRegions)
        self.nHRs = len(self.HeldoutRegions)
        # note that this model will always predict ALL days, setting the other ones to observe only

        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.HR_indxs = [self.ORs.index(r) for r in self.HeldoutRegions]
        self.OR_indxs = list(range(self.nRs))
        [self.ORs.remove(hr) for hr in self.HeldoutRegions]
        [self.OR_indxs.pop(hr_indx) for hr_indx in self.HR_indxs]

    def build_reduction_lognorm(self, scale=0.1):
        self.LN("CMReduction", 0, scale, shape=(self.nCMs,))

    def build_reduction_exp_gamma(self, alpha=0.5, beta=1.0):
        self.CMReductionGamma = pm.Gamma(
            "CMReductionGamma", alpha, beta, shape=(self.nCMs,)
        )
        self.Det("CMReduction", T.exp((-1.0) * self.CMReductionGamma))

    def build_heldout_region_priors(
        self,
        init_size_dist,
        init_size_kwargs,
        growth_rate_dist,
        growth_rate_kwargs,
        region_scale_dist,
        region_scale_kwargs,
    ):
        self.HeldoutInitialSize = init_size_dist(
            name="HeldoutInitialSize", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate", shape=(self.nHRs), **growth_rate_kwargs
        )
        self.HeldoutRegionScales = region_scale_dist(
            name="HeldoutRegionScale", shape=(self.nHRs), **region_scale_kwargs
        )

    def build_rates(self):
        self.LN("BaseGrowthRate", np.log(1.2), 2.0)
        self.LN(
            "RegionGrowthRate",
            pm.math.log(self.BaseGrowthRate),
            0.3,
            shape=(self.nORs,),
        )
        self.LN("RegionScaleMult", 0.0, 1.0, shape=(self.nORs,), plot_trace=False)
        self.ActiveCMReduction = (
            T.reshape(self.CMReduction, (1, self.nCMs, 1))
            ** self.d.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.prod(self.ActiveCMReduction, axis=1), plot_trace=False
        )
        self.DelayedGrowthReduction = geom_convolution(
            self.GrowthReduction, self.DelayProb, axis=1
        )[:, self.CMDelayCut :]
        self.Det(
            "PredictedGrowth",
            T.reshape(self.RegionGrowthRate, (self.nORs, 1))
            * self.DelayedGrowthReduction,
            plot_trace=False,
        )
        self.LN(
            "DailyGrowth",
            pm.math.log(self.PredictedGrowth),
            self.RegionScaleMult.reshape((self.nORs, 1))
            * self.DailyGrowthNoiseMultiplier,
            shape=(self.nORs, self.nDs - self.CMDelayCut),
            plot_trace=False,
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutGrowthReduction = (
                T.reshape(self.CMReduction, (1, self.nCMs, 1))
                ** self.d.ActiveCMs[self.HR_indxs, :]
            )
            self.Det(
                "HeldoutGrowthReduction",
                T.prod(self.HeldoutGrowthReduction, axis=1),
                plot_trace=False,
            )
            self.DelayedGrowthReduction = geom_convolution(
                self.HeldoutGrowthReduction, self.DelayProb, axis=1
            )[:, self.CMDelayCut :]
            self.Det(
                "HeldoutPredictedGrowth",
                T.reshape(self.HeldoutGrowthRate, (self.nHRs, 1))
                * self.DelayedGrowthReduction,
                plot_trace=False,
            )
            self.LN(
                "HeldoutDailyGrowth",
                pm.math.log(self.HeldoutPredictedGrowth),
                self.HeldoutRegionScales.reshape((self.nHRs, 1))
                * self.DailyGrowthNoiseMultiplier,
                shape=(self.nHRs, self.nDs - self.CMDelayCut),
                plot_trace=False,
            )

    def build_output_model(self):
        self.LN("InitialSize", 0, 10, shape=(self.nORs,))
        self.Det(
            "ObservedSize",
            T.reshape(self.InitialSize, (self.nORs, 1))
            * self.DailyGrowth.cumprod(axis=1),
            plot_trace=False,
        )
        self.ObservedLN(
            "Observed",
            pm.math.log(
                self.ObservedSize[:, (self.ObservedDaysIndx - self.CMDelayCut)]
            ),
            self.RegionScaleMult.reshape((self.nORs, 1))
            * self.ConfirmedCasesNoiseMultiplier,
            shape=(self.nORs, self.nODs),
            observed=self.d.Confirmed[self.OR_indxs, :][
                :, self.ObservedDaysIndx
            ],  # ugly, sadly
            plot_trace=False,
        )

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutSize",
                T.reshape(self.HeldoutInitialSize, (self.nHRs, 1))
                * self.HeldoutDailyGrowth.cumprod(axis=1),
                plot_trace=False,
            )
            self.LN(
                "HeldoutObserved",
                pm.math.log(self.HeldoutSize),
                self.HeldoutRegionScales.reshape((self.nHRs, 1))
                * self.ConfirmedCasesNoiseMultiplier,
                shape=(self.nHRs, self.nDs - self.CMDelayCut),
                plot_trace=False,
            )

    def build_all(self):
        self.build_reduction_lognorm()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")
