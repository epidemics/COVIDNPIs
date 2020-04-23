import copy
import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pymc3 as pm
import theano.tensor as T
from pymc3 import Model

from epimodel.pymc3_models.utils import geom_convolution, convolution

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt


def save_fig_pdf(output_dir, figname):
    datetime_str = datetime.now().strftime("%d-%m;%H-%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(f"Saving Plot at {os.path.abspath(output_dir)}")
    plt.savefig(
        f"{output_dir}/{figname}_t{datetime_str}.pdf"
    )


def produce_CIs(data):
    means = np.mean(data, axis=0)
    li = np.percentile(data, 2.5, axis=0)
    ui = np.percentile(data, 97.5, axis=0)
    err = np.array([means - li, ui - means])
    return means, li, ui, err


class BaseCMModel(Model):
    def __init__(self, data, heldout_days=None, heldout_regions=None, name="", model=None):
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

    def Normal(self, name, mean, sigma, plot_trace=True, hyperprior=None, shape=None):
        """Create a lognorm variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        if hyperprior:
            # TODO
            pass
        kws = {}
        if shape is not None:
            kws["shape"] = shape
        v = pm.Normal(name, mean, sigma, **kws)
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

    def plot_region_predictions(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):
            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            ax = plt.gca()
            means, li, ui, err = produce_CIs(np.exp(self.trace.Infected_log[:, country_indx, :]))
            means_delayed, li_delayed, ui_delayed, err_delayed = produce_CIs(
                self.trace.ExpectedConfirmed[:, country_indx, :])
            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            if self.nHODs > 0:
                means_ho, li_ho, ui_ho, err_ho = produce_CIs(
                    self.trace.HeldoutDaysObserved[:, country_indx, :])

                plt.errorbar(self.HeldoutDaysIndx, means_ho, yerr=err_ho, fmt="-^", linewidth=1, markersize=2,
                             label="Heldout Pred Confirmed", zorder=1)
                plt.scatter(self.HeldoutDaysIndx, labels[self.HeldoutDaysIndx], label="Heldout Confirmed", marker="*",
                            color="tab:red", zorder=3)

            labels = self.d.Active[country_indx, :]

            plt.errorbar(days_x, means, yerr=err, fmt="-D", linewidth=1, markersize=2, label="Infected",
                         zorder=1)
            plt.errorbar(days_x, means_delayed, yerr=err_delayed, fmt="-o", linewidth=1, markersize=2,
                         label="Mean Pred Confirmed", zorder=2)
            plt.scatter(self.ObservedDaysIndx, labels[self.ObservedDaysIndx], label="Observed Confirmed", marker="o",
                        s=6, color="tab:purple",
                        zorder=3)

            plt.scatter(days_x, self.d.Deaths[country_indx, :], label="deaths", marker="+",
                        s=6, color="black",
                        zorder=3)

            plt.scatter(days_x[1:], self.d.Deaths[country_indx, 1:] - self.d.Deaths[country_indx, 0:-1],
                        label="new deaths", marker="*",
                        s=6, color="tab:gray",
                        zorder=3)

            # plot countermeasures
            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]

            features = [
                'Mask wearing over 70%',
                'Gatherings limited to 10',
                'Gatherings limited to 100',
                'Gatherings limited to 1000',
                'Business suspended - some',
                'Business suspended - many',
                'Schools and universities closed',
                'General curfew',
                'Healthcare specialisation']

            colors = ["tab:blue", "black", "gray", "lightgray", "tomato", "darkred", "tab:orange", "tab:green",
                      "tab:purple"]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])
                for c in changes:
                    if c.size > 0:
                        height += 1
                        if CM_changes[cm, c] == 1:
                            plt.plot([c, c], [0, 10 ** 6], "--g", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text((c - min_x) / (max_x - min_x), 1 - (0.035 * (height)), f"{cm + 1}",
                                     color=colors[cm],
                                     transform=ax.transAxes, fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor=colors[cm], boxstyle='round'))
                        else:
                            plt.plot([c, c], [0, 10 ** 6], "--r", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text((c - min_x) / (max_x - min_x), 1 - (0.035 * (height)), f"{cm + 1}",
                                     color=colors[cm],
                                     transform=ax.transAxes, fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor=colors[cm], boxstyle='round'))

            ax.set_yscale("log")
            plt.plot([0, 10 ** 6], [0, 10 ** 6], "-r")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -2, 10 ** 6])
            plt.title(f"Region {region}")

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax1 = plt.gca()
            means_growth, _, _, err = produce_CIs(self.trace.ExpectedGrowth[:, country_indx, :])
            actual_growth, _, _, err_act = produce_CIs(self.trace.Growth[:, country_indx, :])

            first_day = np.min(self.ObservedDaysIndx)
            plt.errorbar(days_x, np.exp(actual_growth), yerr=err_act, fmt="-x", linewidth=1, markersize=2,
                         label="Predicted Growth",
                         zorder=1, color="tab:orange")
            plt.errorbar(days_x, np.exp(means_growth), yerr=err, fmt="-D", linewidth=1, markersize=2,
                         label="Expected Growth",
                         zorder=2, color="tab:blue")

            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])
                for c in changes:
                    if c.size > 0:
                        height += 1
                        if CM_changes[cm, c] == 1:
                            plt.plot([c, c], [0, 2], "--g", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text(c, 2 - (0.05 * (height)), f"{cm + 1}", color="g",
                                     fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor='g', boxstyle='round'))
                        else:
                            plt.plot([c, c], [0, 2], "--r", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text(c, 2 - (0.05 * (height)), f"{cm + 1}", color="r",
                                     fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor='g', boxstyle='round'))
            plt.ylim([0.8, 2])
            plt.xlim([first_day, 66])

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            ax2 = plt.gca()
            z1_mean, _, _, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, _, _, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            plt.errorbar(days_x, z1_mean, yerr=err_1, fmt="-x", linewidth=1, markersize=2,
                         label="Growth Noise",
                         zorder=1, color="tab:blue")
            plt.errorbar(self.ObservedDaysIndx, z2_mean, yerr=err_2, fmt="-x", linewidth=1, markersize=2,
                         label="Confirmation Noise",
                         zorder=1, color="tab:orange")
            plt.xlim([first_day, 66])

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(output_dir, f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}")

            elif country_indx % 5 == 0:
                ax.legend(prop={'size': 6})
                ax1.legend(prop={'size': 6})
                ax2.legend(prop={'size': 6})

    def plot_effect(self, save_fig=True, output_dir="./out", x_min=0.5, x_max=1.5):
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

        plt.xlim([x_min, x_max])
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
        if save_fig:
            save_fig_pdf(output_dir, f"CMEffect")

    def run(self, N, chains=2, cores=2):
        print(self.check_test_point())
        with self.model:
            self.trace = pm.sample(N, chains=chains, cores=cores, init="adapt_diag")

    def heldout_days_validation_plot(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None
        assert self.HeldoutDays is not None

        for indx, ho_day in enumerate(self.HeldoutDays):
            labels = self.d.Active[:, ho_day]
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
                save_fig_pdf(output_dir, f"HeldoutDaysValidation_d{ho_day}")

    def heldout_regions_validation_plot(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None
        assert self.HeldoutRegions is not None

        for indx, region_indx in enumerate(self.HR_indxs):
            days = self.d.Ds
            days_x = np.arange(len(days))
            labels = self.d.Active[region_indx, days_x]
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
                save_fig_pdf(output_dir, f"HeldoutRegionValidation_r{self.HeldoutRegions[indx]}")


class CMModelFlexibleV1(BaseCMModel):
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
        self.predict_all_days = False

    def build_cm_reduction_prior(self, dist=None, dist_kwarg=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.CM_Alpha = dist(name="CM_Alpha", **dist_kwarg, shape=(self.nCMs,))
        else:
            with self.model:
                self.CM_Alpha = pm.Gamma(
                    "CM_Alpha", 0.5, 1, shape=(self.nCMs,)
                )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

        if plot_trace:
            self.plot_trace_vars.add("CMReduction")

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.Normal("RegionGrowthRate_log", np.log(1.2), 0.3, shape=(self.nORs,))

    def build_heldout_region_priors(
            self,
            init_size_dist,
            init_size_kwargs,
            growth_rate_dist,
            growth_rate_kwargs
    ):
        self.HeldoutInitialSize = init_size_dist(
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )

        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

    def build_rates(self, growth_noise_dist=None, growth_noise_kwargs=None, transform_mean_lambda=None):

        if transform_mean_lambda is None:
            transform_mean_lambda = lambda x: x

        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "DelayedGrowthReduction",
            convolution(T.sum(self.ActiveCMReduction, axis=1), self.DelayProb, axis=1)[:, self.CMDelayCut:],
            plot_trace=False
        )

        if growth_noise_dist is not None:
            with self.model:
                self.Growth = growth_noise_dist(
                    name="Growth_log",
                    mu=transform_mean_lambda(
                        T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.DelayedGrowthReduction),
                    shape=(self.nORs, self.nDs),
                    **growth_noise_kwargs)
        else:
            self.Normal(
                "Growth_log",
                T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.DelayedGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "DelayedHeldoutGrowthReduction",
                convolution(T.sum(self.HeldoutActiveCMReduction, axis=1), self.DelayProb, axis=1)[:, self.CMDelayCut:],
                plot_trace=False,
            )

            if growth_noise_dist is not None:
                with self.model:
                    self.HeldoutGrowth = growth_noise_dist(
                        name="HeldoutGrowth_log",
                        mu=transform_mean_lambda(
                            T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.DelayedHeldoutGrowthReduction),
                        shape=(self.nHRs, self.nDs),
                        **growth_noise_kwargs)
            else:
                self.LN(
                    "HeldoutGrowth_log",
                    T.reshape(self.HeldoutGrowthRate, (self.nHRs, 1)) * self.DelayedHeldoutGrowthReduction,
                    self.DailyGrowthNoise,
                    shape=(self.nHRs, self.nDs),
                    plot_trace=False,
                )

    def build_output_model(self):
        self.Normal("InitialSize_log", 0, 10, shape=(self.nORs,))

        self.Det(
            "ExpectedConfirmed_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth_log.cumsum(axis=1),
            plot_trace=False
        )

        self.Det("ExpectedConfirmed", pm.math.exp(self.ExpectedConfirmed_log))

        with self.model:
            self.Observed_log = pm.Normal("Observed_log", self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
                                          self.ConfirmationNoise, shape=(self.nORs, self.nODs),
                                          observed=np.log(self.d.Active[self.OR_indxs, :][
                                                          :, self.ObservedDaysIndx
                                                          ]))

        self.Det("Observed", pm.math.exp(self.Observed_log))

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise,
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

        self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                * self.HeldoutGrowth.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed,
                self.ConfirmationNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")


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
        self.ConfirmationNoise = 0.25

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        self.CMReductionGamma = pm.Gamma(
            "CMReductionGamma", alpha, beta, shape=(self.nCMs,)
        )
        self.Det("CMReduction", T.exp((-1.0) * self.CMReductionGamma))

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

        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CMReduction, (1, self.nCMs, 1))
                ** self.ActiveCMs[self.OR_indxs, :]
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
                    ** self.ActiveCMs[self.HR_indxs, :]
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
                    shape=(self.nORs, self.nODs),
                    observed=self.d.Active[self.OR_indxs, :][
                             :, self.ObservedDaysIndx],
                    **confirmed_noise_kwargs)
        else:
            self.ObservedLN(
                "Observed",
                pm.math.log(
                    self.ExpectedConfirmed[:, self.ObservedDaysIndx]
                ),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=self.d.Active[self.OR_indxs, :][
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
                        shape=(self.nORs, self.nHODs),
                        **confirmed_noise_kwargs)
            else:
                self.LN(
                    "HeldoutDaysObserved",
                    pm.math.log(
                        self.ExpectedConfirmed[:, self.HeldoutDaysIndx]
                    ),
                    self.ConfirmationNoise,
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
                        shape=(self.nHRs, self.nDs),
                        **confirmed_noise_kwargs)
            else:
                self.LN(
                    "HeldoutConfirmed",
                    pm.math.log(
                        self.HeldoutExpectedConfirmed + 1e-6
                    ),
                    self.ConfirmationNoise,
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


class CMModelFlexibleV3p1(BaseCMModel):
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
        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.4

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        with self.model:
            self.CM_Alpha = pm.Gamma(
                "CM_Alpha", alpha, beta, shape=(self.nCMs,)
            )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.Normal("RegionGrowthRate_log", np.log(1.2), 0.3, shape=(self.nORs,))

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.Normal("RegionNoiseScale", 1.0, 0.05, shape=(self.nORs,))

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
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self):
        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
        )

        self.Det(
            "ExpectedGrowth",
            T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.GrowthReduction,
            plot_trace=False,
        )

        self.Normal(
            "Growth",
            self.ExpectedGrowth,
            self.DailyGrowthNoise,
            shape=(self.nORs, self.nDs),
            plot_trace=False
        )

        self.Det(
            "Z1",
            self.Growth - self.ExpectedGrowth,
            plot_trace=False
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "HeldoutGrowthReduction",
                T.sum(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            self.Normal(
                "HeldoutGrowth_log",
                T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.HeldoutGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

    def build_output_model(self):
        self.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
        self.Det(
            "Infected_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth.cumsum(axis=1),
            plot_trace=False
        )

        self.Det("Infected", pm.math.exp(self.Infected_log), plot_trace=False)

        # use the theano convolution function, reshaping as required
        expected_confirmed_log = T.nnet.conv2d(self.Infected_log.reshape((1, 1, self.nORs, self.nDs)),
                                               np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                                               border_mode="full")[:, :, :, :self.nDs]
        self.Det("ExpectedConfirmed_log", expected_confirmed_log.reshape((self.nORs, self.nDs)), plot_trace=False)
        self.Det("ExpectedConfirmed", pm.math.exp(self.ExpectedConfirmed_log), plot_trace=False)

        with self.model:
            self.Observed_log = pm.Normal("Observed_log", self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
                                          self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                                          shape=(self.nORs, self.nODs),
                                          observed=np.log(self.d.Active[self.OR_indxs, :][
                                                          :, self.ObservedDaysIndx
                                                          ]))

        self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
        self.Det(
            "Z2",
            self.Observed_log - self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
            plot_trace=False
        )

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

            self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                + self.HeldoutGrowth_log.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutInfected", pm.math.exp(self.HeldoutInfected_log), plot_trace=False)

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed_log = T.nnet.conv2d(
                self.HeldoutInfected_log.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed_log,
                self.ConfirmationNoise * T.reshape(self.HeldoutNoiseScale, (self.nHRs, 1)),
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")


class CMModelFlexibleV3p2(BaseCMModel):
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
        self.DailyGrowthNoise = 0.05
        self.ConfirmationNoise = 0.2

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        with self.model:
            self.CM_Alpha = pm.Gamma(
                "CM_Alpha", alpha, beta, shape=(self.nCMs,)
            )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.Normal("RegionGrowthRate_log", np.log(1.2), 0.3, shape=(self.nORs,))

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.Normal("RegionNoiseScale", 1.0, 0.05, shape=(self.nORs,))

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
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self):
        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.nnet.sigmoid(T.sum(self.ActiveCMReduction, axis=1)), plot_trace=False
        )

        self.Det(
            "ExpectedGrowth",
            T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.GrowthReduction,
            plot_trace=False,
        )

        self.Normal(
            "Growth",
            self.ExpectedGrowth,
            self.DailyGrowthNoise,
            shape=(self.nORs, self.nDs),
            plot_trace=False
        )

        self.Det(
            "Z1",
            self.Growth - self.ExpectedGrowth,
            plot_trace=False
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "HeldoutGrowthReduction",
                T.sum(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            self.Normal(
                "HeldoutGrowth_log",
                T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.HeldoutGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

    def build_output_model(self):
        self.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
        self.Det(
            "Infected_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth.cumsum(axis=1),
            plot_trace=False
        )

        self.Det("Infected", pm.math.exp(self.Infected_log), plot_trace=False)

        # use the theano convolution function, reshaping as required
        expected_confirmed_log = T.nnet.conv2d(self.Infected_log.reshape((1, 1, self.nORs, self.nDs)),
                                               np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                                               border_mode="full")[:, :, :, :self.nDs]
        self.Det("ExpectedConfirmed_log", expected_confirmed_log.reshape((self.nORs, self.nDs)), plot_trace=False)
        self.Det("ExpectedConfirmed", pm.math.exp(self.ExpectedConfirmed_log), plot_trace=False)

        with self.model:
            self.Observed_log = pm.Normal("Observed_log", self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
                                          self.ConfirmationNoise,
                                          shape=(self.nORs, self.nODs),
                                          observed=np.log(self.d.Active[self.OR_indxs, :][
                                                          :, self.ObservedDaysIndx
                                                          ]))

        self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
        self.Det(
            "Z2",
            self.Observed_log - self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
            plot_trace=False
        )

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise,
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

            self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                + self.HeldoutGrowth.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutInfected", pm.math.exp(self.HeldoutInfected_log), plot_trace=False)

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed_log = T.nnet.conv2d(
                self.HeldoutInfected_log.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed,
                self.ConfirmationNoise * T.reshape(self.HeldoutNoiseScale, (self.nHRs, 1)),
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")


class CMModelFlexibleV3p3(BaseCMModel):
    def __init__(
            self, data, heldout_days=None, heldout_regions=None, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.CMDelayCut = 10
        self.DelayProb = np.array([0,
                                   2.10204045e-06, 3.22312869e-05, 1.84979560e-04, 6.31412913e-04,
                                   1.53949439e-03, 3.07378372e-03, 5.32847235e-03, 8.32057678e-03,
                                   1.19864352e-02, 1.59626950e-02, 2.02752812e-02, 2.47013776e-02,
                                   2.90892369e-02, 3.30827134e-02, 3.66035310e-02, 3.95327745e-02,
                                   4.19039762e-02, 4.35677913e-02, 4.45407357e-02, 4.49607434e-02,
                                   4.47581467e-02, 4.40800885e-02, 4.28367817e-02, 4.10649618e-02,
                                   3.93901360e-02, 3.71499615e-02, 3.48922699e-02, 3.24149652e-02,
                                   3.00269472e-02, 2.76836725e-02, 2.52794388e-02, 2.29349630e-02,
                                   2.07959867e-02, 1.86809336e-02, 1.67279378e-02, 1.50166767e-02,
                                   1.33057159e-02, 1.17490048e-02, 1.03030011e-02, 9.10633952e-03,
                                   7.97333972e-03, 6.95565185e-03, 6.05717970e-03, 5.25950540e-03,
                                   4.61137626e-03, 3.94442886e-03, 3.37948046e-03, 2.91402865e-03,
                                   2.48911619e-03, 2.14007737e-03, 1.81005702e-03, 1.54339818e-03,
                                   1.32068199e-03, 1.11358095e-03, 9.53425490e-04, 7.99876440e-04,
                                   6.76156345e-04, 5.68752088e-04, 4.93278826e-04, 4.08596625e-04,
                                   3.37127249e-04, 2.92283720e-04, 2.41934846e-04, 1.98392580e-04])
        self.DailyGrowthNoise = 0.3
        self.ConfirmationNoise = 0.1

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        with self.model:
            self.CM_Alpha = pm.Gamma(
                "CM_Alpha", alpha, beta, shape=(self.nCMs,)
            )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.Normal("RegionGrowthRate_log", np.log(1.2), 0.3, shape=(self.nORs,))

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.Normal("RegionNoiseScale", 1.0, 0.05, shape=(self.nORs,))

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
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self):
        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
        )

        self.Det(
            "ExpectedGrowth",
            T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.GrowthReduction,
            plot_trace=False,
        )

        self.Normal(
            "Growth",
            self.ExpectedGrowth,
            self.DailyGrowthNoise,
            shape=(self.nORs, self.nDs),
            plot_trace=False
        )

        self.Det(
            "Z1",
            self.Growth - self.ExpectedGrowth,
            plot_trace=False
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "HeldoutGrowthReduction",
                T.sum(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            self.Normal(
                "HeldoutGrowth_log",
                T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.HeldoutGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

    def build_output_model(self):
        self.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
        self.Det(
            "Infected_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth.cumsum(axis=1),
            plot_trace=False
        )

        self.Det("Infected", pm.math.exp(self.Infected_log), plot_trace=False)

        # use the theano convolution function, reshaping as required
        expected_confirmed = T.nnet.conv2d(self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                                           np.reshape(self.DelayProb,
                                                      newshape=(1, 1, 1, self.DelayProb.size)),
                                           border_mode="full")[:, :, :, :self.nDs]

        self.Det("ExpectedConfirmed", expected_confirmed.reshape((self.nORs, self.nDs)), plot_trace=False)
        self.Det("ExpectedConfirmed_log", pm.math.log(self.ExpectedConfirmed),
                 plot_trace=False)

        with self.model:
            self.Observed = pm.Lognormal("Observed", pm.math.log(self.ExpectedConfirmed[:, self.ObservedDaysIndx]),
                                         self.ConfirmationNoise * T.reshape(self.RegionNoiseScale,
                                                                            (self.nORs, 1)),
                                         shape=(self.nORs, self.nODs),
                                         observed=self.d.Deaths[self.OR_indxs, :][
                                                  :, self.ObservedDaysIndx]
                                         )

        # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
        # self.Det(
        #     "Z2",
        #     self.Observed_log - self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
        #     plot_trace=False
        # )

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

            self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                + self.HeldoutGrowth_log.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutInfected", pm.math.exp(self.HeldoutInfected_log), plot_trace=False)

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed_log = T.nnet.conv2d(
                self.HeldoutInfected_log.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed_log,
                self.ConfirmationNoise * T.reshape(self.HeldoutNoiseScale, (self.nHRs, 1)),
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")

    def plot_region_predictions(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):
            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            ax = plt.gca()
            means, li, ui, err = produce_CIs(np.exp(self.trace.Infected_log[:, country_indx, :]))
            means_delayed, li_delayed, ui_delayed, err_delayed = produce_CIs(
                self.trace.ExpectedConfirmed[:, country_indx, :])
            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            if self.nHODs > 0:
                means_ho, li_ho, ui_ho, err_ho = produce_CIs(
                    self.trace.HeldoutDaysObserved[:, country_indx, :])

                plt.errorbar(self.HeldoutDaysIndx, means_ho, yerr=err_ho, fmt="-^", linewidth=1, markersize=2,
                             label="Heldout Pred Confirmed", zorder=1)
                plt.scatter(self.HeldoutDaysIndx, labels[self.HeldoutDaysIndx], label="Heldout Confirmed", marker="*",
                            color="tab:red", zorder=3)

            labels = self.d.Deaths[country_indx, :]

            plt.errorbar(days_x, means, yerr=err, fmt="-D", linewidth=1, markersize=2, label="Infected",
                         zorder=1)
            plt.errorbar(days_x, means_delayed, yerr=err_delayed, fmt="-o", linewidth=1, markersize=2,
                         label="Mean Pred Confirmed", zorder=2)
            plt.scatter(self.ObservedDaysIndx, labels[self.ObservedDaysIndx], label="Observed Confirmed", marker="o",
                        s=6, color="tab:purple",
                        zorder=3)

            # plot countermeasures
            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])
                for c in changes:
                    if c.size > 0:
                        height += 1
                        if CM_changes[cm, c] == 1:
                            plt.plot([c, c], [0, 10 ** 6], "--g", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text((c - min_x) / (max_x - min_x), 1 - (0.035 * (height)), f"{cm + 1}", color="g",
                                     transform=ax.transAxes, fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor='g', boxstyle='round'))
                        else:
                            plt.plot([c, c], [0, 10 ** 6], "--r", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text((c - min_x) / (max_x - min_x), 1 - (0.035 * (height)), f"{cm + 1}", color="r",
                                     transform=ax.transAxes, fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor='g', boxstyle='round'))

            ax.set_yscale("log")
            plt.plot([0, 10 ** 6], [0, 10 ** 6], "-r")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -2, 10 ** 6])
            plt.title(f"Region {region}")

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax1 = plt.gca()
            means_growth, _, _, err = produce_CIs(self.trace.ExpectedGrowth[:, country_indx, :])
            actual_growth, _, _, err_act = produce_CIs(self.trace.Growth[:, country_indx, :])

            first_day = np.min(self.ObservedDaysIndx)
            plt.errorbar(days_x, np.exp(actual_growth), yerr=err_act, fmt="-x", linewidth=1, markersize=2,
                         label="Predicted Growth",
                         zorder=1, color="tab:orange")
            plt.errorbar(days_x, np.exp(means_growth), yerr=err, fmt="-D", linewidth=1, markersize=2,
                         label="Expected Growth",
                         zorder=2, color="tab:blue")

            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])
                for c in changes:
                    if c.size > 0:
                        height += 1
                        if CM_changes[cm, c] == 1:
                            plt.plot([c, c], [0, 2], "--g", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text(c, 2 - (0.05 * (height)), f"{cm + 1}", color="g",
                                     fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor='g', boxstyle='round'))
                        else:
                            plt.plot([c, c], [0, 2], "--r", alpha=0.5, linewidth=1, zorder=-2)
                            plt.text(c, 2 - (0.05 * (height)), f"{cm + 1}", color="r",
                                     fontsize=5, backgroundcolor="white",
                                     horizontalalignment="center", zorder=-1,
                                     bbox=dict(facecolor='white', edgecolor='g', boxstyle='round'))
            plt.ylim([0.8, 2])
            plt.xlim([first_day, 66])

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            ax2 = plt.gca()
            z1_mean, _, _, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            plt.errorbar(days_x, z1_mean, yerr=err_1, fmt="-x", linewidth=1, markersize=2,
                         label="Growth Noise",
                         zorder=1, color="tab:blue")
            plt.xlim([first_day, 66])

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(output_dir, f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}")

            elif country_indx % 5 == 0:
                ax.legend(prop={'size': 6})
                ax1.legend(prop={'size': 6})
                ax2.legend(prop={'size': 6})


class CMModelFlexibleV3p4(BaseCMModel):
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
        self.DailyGrowthNoise = 0.3
        self.ConfirmationNoise = 0.1

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        with self.model:
            self.CM_Alpha = pm.Gamma(
                "CM_Alpha", alpha, beta, shape=(self.nCMs,)
            )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.Normal("RegionGrowthRate_log", np.log(1.2), 0.3, shape=(self.nORs,))

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.Normal("RegionNoiseScale", 1.0, 0.05, shape=(self.nORs,))

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
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self):
        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
        )

        self.Det(
            "ExpectedGrowth",
            T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.GrowthReduction,
            plot_trace=False,
        )

        self.Normal(
            "Growth",
            self.ExpectedGrowth,
            self.DailyGrowthNoise,
            shape=(self.nORs, self.nDs),
            plot_trace=False
        )

        self.Det(
            "Z1",
            self.Growth - self.ExpectedGrowth,
            plot_trace=False
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "HeldoutGrowthReduction",
                T.sum(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            self.Normal(
                "HeldoutGrowth_log",
                T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.HeldoutGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

    def build_output_model(self):
        self.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
        self.Det(
            "Infected_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth.cumsum(axis=1),
            plot_trace=False
        )

        self.Det("Infected", pm.math.exp(self.Infected_log), plot_trace=False)

        # use the theano convolution function, reshaping as required
        expected_confirmed_log = T.nnet.conv2d(self.Infected_log.reshape((1, 1, self.nORs, self.nDs)),
                                               np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                                               border_mode="full")[:, :, :, :self.nDs]
        self.Det("ExpectedConfirmed_log", expected_confirmed_log.reshape((self.nORs, self.nDs)), plot_trace=False)
        self.Det("ExpectedConfirmed", pm.math.exp(self.ExpectedConfirmed_log), plot_trace=False)

        with self.model:
            self.Observed_log = pm.Normal("Observed_log", self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
                                          self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                                          shape=(self.nORs, self.nODs),
                                          observed=np.log(self.d.Active[self.OR_indxs, :][
                                                          :, self.ObservedDaysIndx
                                                          ]))

        self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
        self.Det(
            "Z2",
            self.Observed_log - self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
            plot_trace=False
        )

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

            self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                + self.HeldoutGrowth_log.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutInfected", pm.math.exp(self.HeldoutInfected_log), plot_trace=False)

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed_log = T.nnet.conv2d(
                self.HeldoutInfected_log.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed_log,
                self.ConfirmationNoise * T.reshape(self.HeldoutNoiseScale, (self.nHRs, 1)),
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")


class CMModelFlexibleV3p5(BaseCMModel):
    "convolution change"

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
        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.4

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        with self.model:
            self.CM_Alpha = pm.Gamma(
                "CM_Alpha", alpha, beta, shape=(self.nCMs,)
            )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.Normal("RegionGrowthRate_log", np.log(1.2), 0.3, shape=(self.nORs,))

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.Normal("RegionNoiseScale", 1.0, 0.05, shape=(self.nORs,))

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
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self):
        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
        )

        self.Det(
            "ExpectedGrowth",
            T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.GrowthReduction,
            plot_trace=False,
        )

        self.Normal(
            "Growth",
            self.ExpectedGrowth,
            self.DailyGrowthNoise,
            shape=(self.nORs, self.nDs),
            plot_trace=False
        )

        self.Det(
            "Z1",
            self.Growth - self.ExpectedGrowth,
            plot_trace=False
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "HeldoutGrowthReduction",
                T.sum(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            self.Normal(
                "HeldoutGrowth_log",
                T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.HeldoutGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

    def build_output_model(self):
        self.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
        self.Det(
            "Infected_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth.cumsum(axis=1),
            plot_trace=False
        )

        self.Det("Infected", pm.math.exp(self.Infected_log), plot_trace=False)

        # use the theano convolution function, reshaping as required
        expected_confirmed_log = T.nnet.conv2d(self.Infected_log.reshape((1, 1, self.nORs, self.nDs)),
                                               np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                                               border_mode="full")[:, :, :, :self.nDs]

        self.Det("ExpectedConfirmed_log", expected_confirmed_log.reshape((self.nORs, self.nDs)), plot_trace=False)
        self.Det("ExpectedConfirmed", T.nnet.conv2d(np.exp(self.Infected_log.reshape((1, 1, self.nORs, self.nDs))),
                                                    np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                                                    border_mode="full")[:, :, :, :self.nDs], plot_trace=False)

        with self.model:
            self.Observed_log = pm.Normal("Observed_log", self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
                                          self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                                          shape=(self.nORs, self.nODs),
                                          observed=np.log(self.d.Active[self.OR_indxs, :][
                                                          :, self.ObservedDaysIndx
                                                          ]))

        self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
        self.Det(
            "Z2",
            self.Observed_log - self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
            plot_trace=False
        )

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

            self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                + self.HeldoutGrowth_log.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutInfected", pm.math.exp(self.HeldoutInfected_log), plot_trace=False)

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed_log = T.nnet.conv2d(
                self.HeldoutInfected_log.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed_log,
                self.ConfirmationNoise * T.reshape(self.HeldoutNoiseScale, (self.nHRs, 1)),
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")


class CMModelFlexibleV4(BaseCMModel):
    def __init__(
            self, data, heldout_days=None, heldout_regions=None, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.CMDelayCut = 10

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

    def build_cm_reduction_exp_gamma_prior(self, alpha=0.5, beta=1.0):
        with self.model:
            self.CM_Alpha = pm.Gamma(
                "CM_Alpha", alpha, beta, shape=(self.nCMs,)
            )

        self.Det("CMReduction", T.exp((-1.0) * self.CM_Alpha))

    def build_region_growth_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionGrowthRate = dist(name="RegionGrowthRate", **dist_kwargs, shape=(self.nORs,))
        else:
            self.HyperGrowthRateMean_log = pm.HalfStudentT("HyperGrowthRateMean_log", nu=10, sigma=np.log(1.2))
            self.HyperGrowthRateVar = pm.HalfStudentT("HyperGrowthRateVar", nu=10, sigma=.3)

            self.Normal("RegionGrowthRate_log", self.HyperGrowthRateMean_log, self.HyperGrowthRateVar,
                        shape=(self.nORs,))

    def build_region_reliability_prior(self, dist=None, dist_kwargs=None, plot_trace=True):
        if dist is not None:
            with self.model:
                self.RegionNoiseScale = dist(name="RegionNoiseScale", shape=(self.nORs,), **dist_kwargs)
        else:
            self.Normal("RegionNoiseScale", 1.0, 0.05, shape=(self.nORs,))

        if plot_trace:
            self.plot_trace_vars.add("RegionNoiseScale")

    def build_noise_scale_prior(self):
        with self.model:
            # self.DailyGrowthNoise = pm.Gamma("DailyGrowthNoise", mu=0.2, sigma=0.1)
            # self.ConfirmationNoise = pm.Gamma("ConfirmationNoise", mu=0.2, sigma=0.1)
            self.DailyGrowthNoise = 0.3
            self.ConfirmationNoise = 0.3

    def build_delay_prior(self):
        with self.model:
            self.HyperDelayPriorMean = pm.Normal("HyperDelayPriorMean", sigma=0.5, mu=12.5)
            self.HyperGrowthRateAlpha = pm.Normal("HyperDelayPriorAlpha", sigma=1.5, mu=6.5)
            self.DelayDist = pm.NegativeBinomial.dist(mu=self.HyperDelayPriorMean, alpha=self.HyperGrowthRateAlpha)

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
            name="HeldoutInitialSize_log", shape=(self.nHRs), **init_size_kwargs
        )
        self.HeldoutGrowthRate = growth_rate_dist(
            name="HeldoutGrowthRate_log", shape=(self.nHRs), **growth_rate_kwargs
        )

        self.HeldoutNoiseScale = noise_scale_dist(
            name="HeldoutNoiseScale", shape=(self.nHRs), **noise_scale_kwargs
        )

    def build_rates(self):
        with self.model:
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

        self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                * self.ActiveCMs[self.OR_indxs, :]
        )
        self.Det(
            "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
        )

        self.Det(
            "ExpectedGrowth",
            T.reshape(self.RegionGrowthRate_log, (self.nORs, 1)) - self.GrowthReduction,
            plot_trace=False,
        )

        self.Normal(
            "Growth",
            self.ExpectedGrowth,
            self.DailyGrowthNoise,
            shape=(self.nORs, self.nDs),
            plot_trace=False
        )

        self.Det(
            "Z1",
            self.Growth - self.ExpectedGrowth,
            plot_trace=False
        )

        # we already have the rates for heldout days, we just need to sort out heldout regions
        if self.nHRs > 0:
            self.HeldoutActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.HR_indxs, :]
            )

            self.Det(
                "HeldoutGrowthReduction",
                T.sum(self.HeldoutActiveCMReduction, axis=1),
                plot_trace=False,
            )

            self.Normal(
                "HeldoutGrowth_log",
                T.reshape(self.HeldoutGrowthRate_log, (self.nHRs, 1)) - self.HeldoutGrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

    def build_output_model(self):
        self.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
        self.Det(
            "Infected_log",
            T.reshape(self.InitialSize_log, (self.nORs, 1))
            + self.Growth.cumsum(axis=1),
            plot_trace=False
        )

        x = np.arange(0, 21)
        delay_prob = pm.math.exp(self.DelayDist.logp(x))
        delay_prob = delay_prob / T.sum(delay_prob)

        self.Det("Infected", pm.math.exp(self.Infected_log), plot_trace=False)

        # use the theano convolution function, reshaping as required
        expected_confirmed = T.nnet.conv2d(self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                                           T.reshape(delay_prob, newshape=(1, 1, 1, delay_prob.size)),
                                           border_mode="full")[:, :, :, :self.nDs].reshape((self.nORs, self.nDs))
        self.Det("ExpectedConfirmed_log", pm.math.log(expected_confirmed), plot_trace=False)
        self.Det("ExpectedConfirmed", expected_confirmed, plot_trace=False)

        with self.model:
            self.Observed = pm.Lognormal("Observed", pm.math.log(self.ExpectedConfirmed[:, self.ObservedDaysIndx]),
                                         self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                                         shape=(self.nORs, self.nODs),
                                         observed=np.log(self.d.Active[self.OR_indxs, :][
                                                         :, self.ObservedDaysIndx
                                                         ]))

        self.Det("Observed_log", pm.math.log(self.Observed), plot_trace=False)
        self.Det(
            "Z2",
            self.Observed_log - self.ExpectedConfirmed_log[:, self.ObservedDaysIndx],
            plot_trace=False
        )

        if len(self.HeldoutDays) > 0:
            self.Normal(
                "HeldoutDaysObserved_log",
                self.ExpectedConfirmed_log[:, self.HeldoutDaysIndx],
                self.ConfirmationNoise * T.reshape(self.RegionNoiseScale, (self.nORs, 1)),
                shape=(self.nORs, self.nHODs),
                plot_trace=False)

            self.Det("HeldoutDaysObserved", pm.math.exp(self.HeldoutDaysObserved_log))

        # we've added observations for observed days for observed regions. need to compute observations for the heldout
        # regions
        if self.nHRs > 0:
            self.Det(
                "HeldoutInfected_log",
                T.reshape(self.HeldoutInitialSize_log, (self.nHRs, 1))
                + self.HeldoutGrowth_log.cumsum(axis=1),
                plot_trace=False
            )

            self.Det("HeldoutInfected", pm.math.exp(self.HeldoutInfected_log), plot_trace=False)

            # use the theano convolution function, reshaping as required
            ho_expected_confirmed_log = T.nnet.conv2d(
                self.HeldoutInfected_log.reshape((1, 1, self.nHRs, self.nDs)),
                np.reshape(self.DelayProb, (1, 1, 1, self.DelayProb.size)),
                border_mode="full")[:, :, :, :self.nDs]
            self.Det("HeldoutExpectedConfirmed_log", ho_expected_confirmed_log.reshape((self.nHRs, self.nDs)),
                     plot_trace=False)

            self.Det("HeldoutExpectedConfirmed", pm.math.exp(self.HeldoutExpectedConfirmed_log))

            self.Normal(
                "HeldoutConfirmed_log",
                self.HeldoutExpectedConfirmed_log,
                self.ConfirmationNoise * T.reshape(self.HeldoutNoiseScale, (self.nHRs, 1)),
                shape=(self.nHRs, self.nDs),
                plot_trace=False,
            )

            self.Det("HeldoutConfirmed", pm.math.exp(self.HeldoutConfirmed_log))

    def build_all(self):
        self.build_cm_reduction_prior()
        self.build_region_growth_prior()
        self.build_region_reliability_prior()
        self.build_rates()
        self.build_output_model()
        log.info("Checking model test point")
        log.info(f"\n{self.check_test_point()}\n")
