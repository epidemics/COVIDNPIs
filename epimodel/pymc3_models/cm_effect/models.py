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

    def plot_observed_region(self, region_code):
        assert self.trace is not None
        region_indx = self.d.Rs.index(region_code)

        self.trace[""]

    def plot_effect(self):
        assert self.trace is not None
        fig = plt.figure(figsize=(7, 3), dpi=300)
        means = np.mean(self.trace["CMReduction"], axis=0)
        li = np.percentile(self.trace["CMReduction"], 2.5, axis=0)
        ui = np.percentile(self.trace["CMReduction"], 97.5, axis=0)
        lq = np.percentile(self.trace["CMReduction"], 25, axis=0)
        uq = np.percentile(self.trace["CMReduction"], 75, axis=0)

        N_cms = means.size

        plt.subplot(121)
        plt.plot([1, 1], [1, -(N_cms)], "--r", linewidth=0.5)
        y_vals = -1 * np.arange(N_cms)
        plt.scatter(means, y_vals, marker="|", color='k')
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.5)

        plt.xlim([0.5, 1.5])
        plt.ylim([-(N_cms - 0.5), 0.5])
        plt.ylabel("Countermeasure", rotation=90)
        plt.yticks(y_vals, [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.xlabel("Countermeasure Effectiveness")

        plt.subplot(122)
        correlation = np.corrcoef(self.trace["CMReduction"], rowvar=False)
        plt.imshow(correlation, cmap="PuOr", vmin=-1, vmax=1)
        plt.colorbar()
        plt.yticks(np.arange(N_cms), [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.xticks(np.arange(N_cms), [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.title("Correlation")

        plt.tight_layout()

    def run(self, N, chains=2, cores=2):
        print(self.check_test_point())
        with self.model:
            self.trace = pm.sample(N, chains=chains, cores=cores, init="adapt_diag")

    def heldout_days_validation_plot(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None
        assert self.HeldoutDays is not None

        for indx, ho_day in enumerate(self.HeldoutDays):
            labels = self.d.Confirmed[:, ho_day]
            predictions = self.trace["HeldoutDaysObserved"][:, :, indx]

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
            predictions = self.trace["HeldoutConfirmed"][:, indx]
            cut_days = self.CMDelayCut

            plt.figure(figsize=(4, 3), dpi=300)
            pred_y = np.mean(predictions, axis=0).flatten()
            li_y = np.percentile(predictions, 2.5, axis=0).T.flatten()
            ui_y = np.percentile(predictions, 97.5, axis=0).T.flatten()
            yerr = np.array([pred_y - li_y, ui_y - pred_y])

            max_val = 10 ** np.ceil(np.log10(max(np.max(ui_y), np.max(labels))))
            min_val = 10 ** np.floor(np.log10(min(np.min(li_y), np.min(labels))))

            if self.predict_all_days:
                plt.errorbar(
                    days_x, np.mean(predictions, axis=0).T, yerr=yerr, zorder=-1
                )
                plt.plot(days_x, labels, "-x", MarkerSize=3, zorder=0)
            else:
                plt.errorbar(
                    days_x[cut_days:], np.mean(predictions, axis=0).T, yerr=yerr, zorder=-1
                )
                plt.plot(days_x[cut_days:], labels[cut_days:], "-x", MarkerSize=3, zorder=0)

            ax = plt.gca()
            ax.set_yscale("log")
            plt.ylim([1, max_val])
            plt.xlim([1, len(days)])
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
            self.HeldoutDaysIndx = np.array(self.HeldoutDays) - self.CMDelayCut
        else:
            self.HeldoutDays = []

        # useful things for heldout stuff
        self.nORs = self.nRs - len(self.HeldoutRegions)
        self.nHRs = len(self.HeldoutRegions)
        self.nHODs = len(self.HeldoutDays)
        # note that this model will always predict ALL days, setting the other ones to observe only

        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.HR_indxs = [self.ORs.index(r) for r in self.HeldoutRegions]
        self.OR_indxs = list(range(self.nRs))
        [self.ORs.remove(hr) for hr in self.HeldoutRegions]
        [self.OR_indxs.pop(hr_indx) for hr_indx in self.HR_indxs]

        # this model predicts but masks early days
        self.predict_all_days = True

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
        )[:, self.CMDelayCut:]

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
            )[:, self.CMDelayCut:]
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
            "InfectionsConfirmed",
            T.reshape(self.InitialSize, (self.nORs, 1))
            * self.DailyGrowth.cumprod(axis=1),
            plot_trace=False,
        )
        self.ObservedLN(
            "Observed",
            pm.math.log(
                self.InfectionsConfirmed[:, (self.ObservedDaysIndx - self.CMDelayCut)]
            ),
            self.RegionScaleMult.reshape((self.nORs, 1))
            * self.ConfirmedCasesNoiseMultiplier,
            shape=(self.nORs, self.nODs),
            observed=self.d.Confirmed[self.OR_indxs, :][
                     :, self.ObservedDaysIndx
                     ],  # ugly, sadly
            plot_trace=False,
        )

        if self.HeldoutDays is not None:
            self.LN(
                "HeldoutDaysObserved",
                pm.math.log(
                    self.InfectionsConfirmed[:, self.HeldoutDaysIndx]
                ),
                self.RegionScaleMult.reshape((self.nORs, 1))
                * self.ConfirmedCasesNoiseMultiplier,
                shape=(self.nORs, self.nHODs),  # ugly, sadly
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
                "HeldoutConfirmed",
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


class CMModelFlexibleV3(BaseCMModel):
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
        self.DailyGrowthNoise = 0.075
        self.ConfirmedCasesNoise = 0.3

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
            self.HeldoutDaysIndx = np.array(self.HeldoutDays)
        else:
            self.HeldoutDays = []
            self.HeldoutDaysIndx = np.array([])

        # useful things for heldout stuff
        self.nORs = self.nRs - len(self.HeldoutRegions)
        self.nHRs = len(self.HeldoutRegions)
        # note that this model will always predict ALL days but heldout ones, setting the other ones to observe only
        self.nHODs = len(self.HeldoutDays)

        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.HR_indxs = [self.ORs.index(r) for r in self.HeldoutRegions]
        self.OR_indxs = list(range(self.nRs))
        [self.ORs.remove(hr) for hr in self.HeldoutRegions]
        [self.OR_indxs.pop(hr_indx) for hr_indx in self.HR_indxs]

        # this model predicts but masks early days
        self.predict_all_days = True

    def build_cm_reduction_prior(self, dist=None, dist_kwarg=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.CMReduction = dist(name="CMReduction", **dist_kwarg, shape=(self.nCMs,))
        else:
            # default to log norm prior
            # log(1) = 0
            # this dist has mean 1
            self.LN("CMReduction", 0, 0.5, shape=(self.nCMs,))

        if plot_trace:
            self.plot_trace_vars.add("CMReduction")

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            # default to log norm prior
            self.LN("RegionGrowthRate", np.log(1.2), 0.4, shape=(self.nORs,))

        if plot_trace:
            self.plot_trace_vars.add("CMReduction")

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.LN("RegionNoiseScale", 0.0, 1.0, shape=(self.nORs,))

        if plot_trace:
            self.plot_trace_vars.add("RegionNoiseScale")

    def build_heldout_region_priors(
            self,
            init_size_dist,
            init_size_kwargs,
            growth_rate_dist,
            growth_rate_kwargs,
            noise_scale_dist,
            noise_scale_kwargs,
    ):
        self.HeldoutInitialSize = init_size_dist(
            name="HeldoutInitialSize", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self, growth_noise_dist=None, growth_noise_kwargs=None, transform_mean_lambda=None):

        if transform_mean_lambda is None:
            transform_mean_lambda = lambda x: x

        self.ActiveCMReduction = (
                T.reshape(self.CMReduction, (1, self.nCMs, 1))
                ** self.d.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.prod(self.ActiveCMReduction, axis=1), plot_trace=False
        )

        if growth_noise_dist is not None:
            with self.model:
                self.Growth = growth_noise_dist(
                    name="Growth",
                    mu=transform_mean_lambda(
                        T.reshape(self.RegionGrowthRate, (self.nORs, 1)) * self.GrowthReduction),
                    # sigma=self.RegionNoiseScale * self.DailyGrowthNoise,
                    shape=(self.nORs, self.nDs),
                    **growth_noise_kwargs)
        else:
            self.LN(
                "Growth",
                pm.math.log(T.reshape(self.RegionGrowthRate, (self.nORs, 1))
                            * self.GrowthReduction),
                self.RegionNoiseScale.reshape((self.nORs, 1)) * self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CMReduction, (1, self.nCMs, 1))
                    ** self.d.ActiveCMs[self.HR_indxs, :]
            )
            self.Det(
                "HeldoutGrowthReduction",
                T.prod(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            if growth_noise_dist is not None:
                with self.Model:
                    self.HeldoutGrowth = growth_noise_dist(
                        name="HeldoutGrowth",
                        mu=transform_mean_lambda(
                            T.reshape(self.HeldoutGrowthRate, (self.nHRs, 1)) * self.HeldoutGrowthReduction),
                        shape=(self.nHRs, self.nDs),
                        **growth_noise_kwargs)
            else:
                self.LN(
                    "HeldoutGrowth",
                    pm.math.log(T.reshape(self.HeldoutGrowthRate, (self.nHRs, 1))
                                * self.HeldoutGrowthReduction),
                    self.HeldoutNoiseScale * self.DailyGrowthNoise,
                    shape=(self.nHRs, self.nDs),
                    plot_trace=False,
                )

    def build_output_model(self, confirmed_noise_dist=None, confirmed_noise_kwargs=None, transform_mean_lambda=None):
        if transform_mean_lambda is None:
            transform_mean_lambda = lambda x: x

        self.LN("InitialSize", 0, 10, shape=(self.nORs,))
        self.Det(
            "Infected",
            T.reshape(self.InitialSize, (self.nORs, 1))
            * self.Growth.cumprod(axis=1),
            plot_trace=False
        )

        # use the theano convolution function, reshaping as required
        expected_confirmed = T.nnet.conv2d(self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                                           np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                                           border_mode="full")[:, :, :, :self.nDs]
        self.Det("ExpectedConfirmed", expected_confirmed.reshape((self.nORs, self.nDs)), plot_trace=False)

        if confirmed_noise_dist is not None:
            with self.model:
                self.Observed = confirmed_noise_dist(
                    name="Observed",
                    mu=transform_mean_lambda(
                        self.ExpectedConfirmed[:, self.ObservedDaysIndx]),
                    sigma=self.RegionNoiseScale * self.ConfirmedCasesNoise,
                    shape=(self.nORs, self.nODs),
                    observed=self.d.Confirmed[self.OR_indxs, :][
                             :, self.ObservedDaysIndx],
                    **confirmed_noise_kwargs)
        else:
            self.ObservedLN(
                "Observed",
                pm.math.log(
                    self.ExpectedConfirmed[:, self.ObservedDaysIndx]
                ),
                #self.RegionNoiseScale.reshape((self.nORs, 1)) * self.ConfirmedCasesNoise,
                0.3,
                shape=(self.nORs, self.nODs),
                observed=self.d.Confirmed[self.OR_indxs, :][
                         :, self.ObservedDaysIndx
                         ],  # ugly, sadly
                plot_trace=False,
            )

        if len(self.HeldoutDays) > 0:
            if confirmed_noise_dist is not None:
                with self.model:
                    self.HeldoutDaysObserved = confirmed_noise_dist(
                        name="HeldoutDaysObserved",
                        mu=transform_mean_lambda(
                            self.ExpectedConfirmed[:, self.HeldoutDaysIndx]),
                        sigma=self.ConfirmedCasesNoise.reshape(self.nORs, 1) * self.RegionNoiseScale,
                        shape=(self.nORs, self.nHODs),
                        **confirmed_noise_kwargs)
            else:
                self.LN(
                    "HeldoutDaysObserved",
                    pm.math.log(
                        self.ExpectedConfirmed[:, self.HeldoutDaysIndx]
                    ),
                    #self.ConfirmedCasesNoise * self.RegionNoiseScale.reshape((self.nORs, 1)),
                    0.3,
                    shape=(self.nORs, self.nHODs),
                    plot_trace=False)

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected",
                T.reshape(self.HeldoutInitialSize, (self.nHRs, 1))
                * self.HeldoutGrowth.cumprod(axis=1),
                plot_trace=False
            )

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed = T.nnet.conv2d(
                self.HeldoutInfected.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            # modify testpoint!
            self.Det("HeldoutExpectedConfirmed", ho_expected_confirmed.reshape((self.nHRs, self.nDs)), plot_trace=False)

            # add jitter to these distributions. we need to if we want the test point to work.
            if confirmed_noise_dist is not None:
                with self.model:
                    self.HeldoutConfirmed = confirmed_noise_dist(
                        name="HeldoutConfirmed",
                        mu=transform_mean_lambda(
                            self.HeldoutExpectedConfirmed + 1e-6),
                        sigma=self.ConfirmedCasesNoise * self.HeldoutNoiseScale.reshape(self.nHRs, 1),
                        shape=(self.nHRs, self.nDs),
                        **confirmed_noise_kwargs)
            else:
                self.LN(
                    "HeldoutConfirmed",
                    pm.math.log(
                        self.HeldoutExpectedConfirmed + 1e-6
                    ),
                    self.ConfirmedCasesNoise * self.HeldoutNoiseScale.reshape((self.nHRs, 1)),
                    shape=(self.nHRs, self.nDs),
                    plot_trace=False,
                )

            # self.HeldoutConfirmed.test_value = np.ones((self.nHRs, self.nDs))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")
