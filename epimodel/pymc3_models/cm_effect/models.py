import copy
import logging
import os
from datetime import datetime

import seaborn as sns

import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano.tensor.signal.conv as C
from pymc3 import Model

log = logging.getLogger(__name__)
sns.set_style("ticks")

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")


def save_fig_pdf(output_dir, figname):
    datetime_str = datetime.now().strftime("%d-%m;%H-%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(f"Saving Plot at {os.path.abspath(output_dir)}")
    plt.savefig(f"{output_dir}/{figname}_t{datetime_str}.pdf")


def produce_CIs(data):
    means = np.mean(data, axis=0)
    li = np.percentile(data, 2.5, axis=0)
    ui = np.percentile(data, 97.5, axis=0)
    err = np.array([means - li, ui - means])
    return means, li, ui, err


def add_cms_to_plot(ax, ActiveCMs, country_indx, min_x, max_x, days, plot_style):
    ax2 = ax.twinx()
    plt.ylim([0, 1])
    plt.xlim([min_x, max_x])
    CMs = ActiveCMs[country_indx, :, :]
    nCMs, _ = CMs.shape
    CM_changes = np.zeros((nCMs, len(days)))
    CM_changes[:, 1:] = CMs[:, 1:] - CMs[:, :-1]
    all_CM_changes = np.sum(CM_changes, axis=0)
    all_heights = np.zeros(all_CM_changes.shape)

    for cm in range(nCMs):
        changes = np.nonzero(CM_changes[cm, :])[0].tolist()
        height = 1
        for c in changes:
            close_heights = all_heights[c - 3:c + 4]
            if len(close_heights) == 7:
                height = np.max(close_heights) + 1
                all_heights[c] = height

            plt.plot(
                [c, c],
                [0, 1],
                "--",
                color="lightgrey",
                linewidth=1,
                zorder=-2,
                alpha=0.5
            )
            plot_height = 1 - (0.04 * height)

            if CM_changes[cm, c] == 1:
                plt.text(c, plot_height, plot_style[cm][0], fontproperties=fp2, color=plot_style[cm][1], size=8,
                         va='center', ha='center', clip_on=True, zorder=1)
            else:
                plt.text(c, plot_height, plot_style[cm][0], fontproperties=fp2, color=plot_style[cm][1], size=8,
                         va='center', ha='center', clip_on=True, zorder=1)
                plt.plot([c - 1.5, c + 1.5], [plot_height - 0.005, plot_height + 0.005], color="black", zorder=2)

    plt.yticks([])
    return ax2


class BaseCMModel(Model):
    def __init__(
            self, data, heldout_days=None, heldout_regions=None, name="", model=None
    ):
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

    def plot_effect(self, save_fig=True, output_dir="./out", x_min=-100, x_max=100):
        assert self.trace is not None
        fig = plt.figure(figsize=(7, 3), dpi=300)
        means = 100 * (1 - np.mean(self.trace["CMReduction"], axis=0))
        li = 100 * (1 - np.percentile(self.trace["CMReduction"], 2.5, axis=0))
        ui = 100 * (1 - np.percentile(self.trace["CMReduction"], 97.5, axis=0))
        lq = 100 * (1 - np.percentile(self.trace["CMReduction"], 25, axis=0))
        uq = 100 * (1 - np.percentile(self.trace["CMReduction"], 75, axis=0))

        N_cms = means.size

        fig = plt.figure(figsize=(4, 3), dpi=300)
        plt.plot([0, 0], [1, -(N_cms)], "--r", linewidth=0.5)
        y_vals = -1 * np.arange(N_cms)
        plt.scatter(means, y_vals, marker="|", color="k")
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.5)

        plt.xlim([x_min, x_max])
        xtick_vals = np.arange(-100, 150, 50)
        xtick_str = [f"{x:.0f}%" for x in xtick_vals]
        plt.ylim([-(N_cms - 0.5), 0.5])
        plt.yticks(y_vals, self.d.CMs, fontsize=6)
        plt.xticks(xtick_vals, xtick_str, fontsize=6)
        plt.xlabel("Percentage Reduction in $R$", fontsize=8)
        sns.despine()

        if save_fig:
            save_fig_pdf(output_dir, f"CMEffect")

        fig = plt.figure(figsize=(7, 3), dpi=300)
        correlation = np.corrcoef(self.trace["CMReduction"], rowvar=False)
        plt.imshow(correlation, cmap="PuOr", vmin=-1, vmax=1)
        cbr = plt.colorbar()
        cbr.ax.tick_params(labelsize=6)
        plt.yticks(np.arange(N_cms), self.d.CMs, fontsize=6)
        plt.xticks(np.arange(N_cms), self.d.CMs, fontsize=6, rotation=90)
        plt.title("Posterior Correlation", fontsize=10)
        sns.despine()

        if save_fig:
            save_fig_pdf(output_dir, f"CMCorr")

    def run(self, N, chains=2, cores=2, **kwargs):
        print(self.check_test_point())
        with self.model:
            self.trace = pm.sample(N, chains=chains, cores=cores, init="adapt_diag", **kwargs)

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


class CMCombined(BaseCMModel):
    def __init__(
            self, data, heldout_days=None, heldout_regions=None, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.CMDelayCut = 10
        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )
        self.DailyGrowthNoise = 0.15
        self.ConfirmationNoise = 0.3
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

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Gamma("CM_Alpha", 0.5, 1, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperGrowthRateMean_log = pm.HalfStudentT(
                "HyperGrowthRateMean_log", nu=10, sigma=np.log(1.2)
            )
            self.HyperGrowthRateVar = pm.HalfStudentT(
                "HyperGrowthRateVar", nu=10, sigma=0.3
            )

            self.RegionGrowthRate_log = pm.Normal("RegionGrowthRate_log", self.HyperGrowthRateMean_log,
                                                  self.HyperGrowthRateVar,
                                                  shape=(self.nORs,))

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
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            # output model - deaths
            self.Phi = pm.HalfNormal(name="Phi", sigma=5)
            self.DeathInitialSize_log = pm.Normal("DeathInitialSize_log", -3, 1, shape=(self.nORs,))
            self.DeathInfected_log = pm.Deterministic("DeathInfected_log", T.reshape(self.DeathInitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected_d = pm.Deterministic("Infected_d", pm.math.exp(self.DeathInfected_log))

            # use the theano convolution function, reshaping as required
            expected_deaths = T.nnet.conv2d(
                self.Infected_d.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",
            )[:, :, :, : self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape((self.nORs, self.nDs)))

            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
                alpha=self.ExpectedDeaths[:, self.ObservedDaysIndx] + self.Phi,
                shape=(self.nORs, self.nODs),
                observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedDeaths - self.ExpectedDeaths[:, self.ObservedDaysIndx],
                plot_trace=False
            )

            # output model - active cases
            self.HyperDelayPriorMean = pm.Normal(
                "HyperDelayPriorMean", sigma=3, mu=12.5
            )
            self.HyperGrowthRateAlpha = pm.Normal(
                "HyperDelayPriorAlpha", sigma=1.5, mu=6.5
            )

            self.ActiveDelayDist = pm.NegativeBinomial.dist(
                mu=self.HyperDelayPriorMean, alpha=self.HyperGrowthRateAlpha
            )

            self.ActiveInitialSize_log = pm.Normal("ActiveInitialSize_log", 1, 10, shape=(self.nORs,))
            self.ActiveInfected_log = pm.Deterministic(
                "ActiveInfected_log",
                T.reshape(self.ActiveInitialSize_log, (self.nORs, 1))
                + self.Growth.cumsum(axis=1),
            )

            x = np.arange(0, 64)
            delay_prob = pm.math.exp(self.ActiveDelayDist.logp(x))
            delay_prob = delay_prob / T.sum(delay_prob)

            self.Infected_a = pm.Deterministic("Infected_a", pm.math.exp(self.ActiveInfected_log))

            # use the theano convolution function, reshaping as required
            expected_confirmed = T.nnet.conv2d(
                self.Infected_a.reshape((1, 1, self.nORs, self.nDs)),
                T.reshape(delay_prob, newshape=(1, 1, 1, delay_prob.size)),
                border_mode="full",
            )[:, :, :, : self.nDs].reshape((self.nORs, self.nDs))

            self.ExpectedActive_log = pm.Deterministic(
                "ExpectedActive_log", pm.math.log(expected_confirmed)
            )

            self.ExpectedActive = pm.Deterministic("ExpectedActive", expected_confirmed)

            self.ObservedActive = pm.Lognormal(
                "ObservedActive",
                pm.math.log(self.ExpectedActive[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=self.d.Active[self.OR_indxs, :][:, self.ObservedDaysIndx],
            )

            self.ObservedActive_log = pm.Deterministic("ObservedActive_log", pm.math.log(self.ObservedActive))
            self.Z3 = pm.Deterministic(
                "Z3",
                self.ObservedActive_log - self.ExpectedActive_log[:, self.ObservedDaysIndx],
            )

    def plot_region_predictions(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):
            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            ax = plt.gca()
            means_d, _, _, err_d = produce_CIs(
                self.trace.Infected_d[:, country_indx, :]
            )

            means_a, _, _, err_a = produce_CIs(
                self.trace.Infected_a[:, country_indx, :]
            )

            means_expected_active, _, _, err_expected_active = produce_CIs(
                self.trace.ExpectedActive[:, country_indx, :]
            )

            means_expected_deaths, _, _, err_expected_deaths = produce_CIs(
                self.trace.ExpectedDeaths[:, country_indx, :]
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]
            active = self.d.Active[country_indx, :]

            if self.nHODs > 0:
                means_ho, li_ho, ui_ho, err_ho = produce_CIs(
                    self.trace.HeldoutDaysObserved[:, country_indx, :]
                )

                plt.errorbar(
                    self.HeldoutDaysIndx,
                    means_ho,
                    yerr=err_ho,
                    fmt="-^",
                    linewidth=1,
                    markersize=2,
                    label="Heldout Pred Confirmed",
                    zorder=1,
                )
                plt.scatter(
                    self.HeldoutDaysIndx,
                    labels[self.HeldoutDaysIndx],
                    label="Heldout Confirmed",
                    marker="*",
                    color="tab:red",
                    zorder=3,
                )

            plt.errorbar(
                days_x,
                means_a,
                yerr=err_a,
                fmt="-D",
                linewidth=1,
                markersize=2,
                label="Infected Active",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.errorbar(
                days_x,
                means_d,
                yerr=err_d,
                fmt="-s",
                linewidth=1,
                markersize=2,
                label="Infected Deaths",
                zorder=1,
                color="tab:red",
                alpha=0.25,
            )

            plt.errorbar(
                days_x,
                means_expected_active,
                yerr=err_expected_active,
                fmt="-o",
                linewidth=1,
                markersize=2,
                label="Expected Active",
                zorder=2,
                color="tab:blue"
            )

            plt.errorbar(
                days_x,
                means_expected_deaths,
                yerr=err_expected_deaths,
                fmt="-o",
                linewidth=1,
                markersize=2,
                label="Expected Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Observed Dead",
                marker="o",
                s=8,
                color="tab:orange",
                alpha=0.75,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Observed Active",
                marker="o",
                s=6,
                color="tab:green",
                zorder=3,
                alpha=0.75
            )

            # plot countermeasures
            colors = ["tab:purple",
                      "tab:blue",
                      "silver",
                      "gray",
                      "black",
                      "tomato",
                      "tab:red",
                      "hotpink",
                      "tab:green"]

            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])[0].tolist()
                for c in changes:
                    height += 1
                    if CM_changes[cm, c] == 1:
                        plt.plot(
                            [c, c],
                            [0, 10 ** 7],
                            color="lightgrey",
                            linewidth=1,
                            zorder=-2,
                        )
                        plt.text(
                            (c - min_x) / (max_x - min_x),
                            1 - (0.035 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            transform=ax.transAxes,
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )
                    else:
                        plt.plot(
                            [c, c],
                            [0, 10 ** 7],
                            color="lightgrey",
                            linewidth=1,
                            zorder=-2,
                        )
                        plt.text(
                            (c - min_x) / (max_x - min_x),
                            1 - (0.035 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            transform=ax.transAxes,
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 7])

            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax1 = plt.gca()
            means_growth, _, _, err = produce_CIs(
                self.trace.ExpectedGrowth[:, country_indx, :]
            )
            actual_growth, _, _, err_act = produce_CIs(
                self.trace.Growth[:, country_indx, :]
            )

            plt.errorbar(
                days_x,
                np.exp(actual_growth),
                yerr=err_act,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Predicted Growth",
                zorder=1,
                color="tab:orange",
            )
            plt.errorbar(
                days_x,
                np.exp(means_growth),
                yerr=err,
                fmt="-D",
                linewidth=1,
                markersize=2,
                label="Expected Growth",
                zorder=2,
                color="tab:blue",
            )

            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])[0].tolist()
                for c in changes:
                    height += 1
                    if CM_changes[cm, c] == 1:
                        plt.plot(
                            [c, c], [0, 2], color="lightgrey", alpha=1, linewidth=1, zorder=-2
                        )
                        plt.text(
                            c,
                            2 - (0.05 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )
                    else:
                        plt.plot(
                            [c, c], [0, 2], "--r", alpha=0.5, linewidth=1, zorder=-2
                        )
                        plt.text(
                            c,
                            2 - (0.05 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )
            plt.ylim([0.7, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax2 = plt.gca()
            z1_mean, _, _, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, _, _, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            z3_mean, _, _, err_3 = produce_CIs(self.trace.Z3[:, country_indx, :])
            ln1 = plt.errorbar(
                days_x,
                z1_mean,
                yerr=err_1,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Growth Noise",
                zorder=1,
                color="tab:blue",
            )

            ln2 = plt.errorbar(
                self.ObservedDaysIndx,
                z3_mean,
                yerr=err_3,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Confirmation Noise",
                zorder=1,
                color="tab:green",
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-axis_scale * (max((np.max(err_3) + np.max(z3_mean)), (np.max(err_1) + np.max(z1_mean)))),
                      axis_scale * ((max((np.max(err_3) + np.max(z3_mean)), (np.max(err_1) + np.max(z1_mean)))))])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("Growth / Confirmation Noise")
            ax3 = plt.twinx()
            ln3 = plt.errorbar(
                self.ObservedDaysIndx,
                z2_mean,
                yerr=err_2,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Death Noise",
                zorder=1,
                color="tab:orange",
            )
            plt.ylabel("Death Noise")

            plt.ylim([-axis_scale * np.max(err_2 + np.max(z2_mean)), axis_scale * (np.max(err_2) + np.max(z2_mean))])
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx % 5 == 0:
                ax.legend(prop={"size": 8})
                ax1.legend(prop={"size": 8})
                ax2.legend(lines + lines2, labels + labels2, prop={"size": 8})

    def plot_effect(self, save_fig=True, output_dir="./out", x_min=0.5, x_max=1.5):
        assert self.trace is not None
        fig = plt.figure(figsize=(7, 3), dpi=300)
        means = np.mean(np.exp(-self.trace["CM_Alpha"]), axis=0)
        li = np.percentile(np.exp(-self.trace["CM_Alpha"]), 2.5, axis=0)
        ui = np.percentile(np.exp(-self.trace["CM_Alpha"]), 97.5, axis=0)
        lq = np.percentile(np.exp(-self.trace["CM_Alpha"]), 25, axis=0)
        uq = np.percentile(np.exp(-self.trace["CM_Alpha"]), 75, axis=0)

        N_cms = means.size

        plt.subplot(121)
        plt.plot([1, 1], [1, -(N_cms)], "--r", linewidth=0.5)
        y_vals = -1 * np.arange(N_cms)
        plt.scatter(means, y_vals, marker="|", color="k")
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.5)

        plt.xlim([x_min, x_max])
        plt.ylim([-(N_cms - 0.5), 0.5])
        plt.ylabel("Countermeasure", rotation=90)
        plt.yticks(y_vals, [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.xlabel("Countermeasure Effectiveness")

        plt.subplot(122)
        correlation = np.corrcoef(np.exp(-self.trace["CM_Alpha"]), rowvar=False)
        plt.imshow(correlation, cmap="PuOr", vmin=-1, vmax=1)
        plt.colorbar()
        plt.yticks(np.arange(N_cms), [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.xticks(np.arange(N_cms), [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.title("Correlation")

        plt.tight_layout()
        sns.despine()
        if save_fig:
            save_fig_pdf(output_dir, f"CMEffect")


class CMDeath(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 10

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperGrowthRateMean_log = pm.HalfStudentT(
                "HyperGrowthRateMean_log", nu=10, sigma=0.3
            )
            self.HyperGrowthRateVar = pm.HalfStudentT(
                "HyperGrowthRateVar", nu=10, sigma=0.3
            )

            self.RegionGrowthRate_log = pm.Normal("RegionGrowthRate_log", self.HyperGrowthRateMean_log,
                                                  self.HyperGrowthRateVar,
                                                  shape=(self.nORs,))

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

            self.DailyGrowthNoise = pm.HalfStudentT("DeathsNoise", nu=10, sigma=.4)

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", -3, 10, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # use the theano convolution function, reshaping as required
            # 1.034649e+00s
            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            # expected_confirmed = convolution(self.Infected, self.DelayProb, axis=1)

            self.DeathNoiseScale = pm.HalfStudentT("DeathsNoise", nu=10, sigma=.4)

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            # self.ObservedDeaths = pm.Normal(
            #     "ObservedDeaths",
            #     mu = self.ExpectedDeaths[:, self.ObservedDaysIndx],
            #     sigma = 0.25 * self.ExpectedDeaths[:, self.ObservedDaysIndx] + 0.25,
            #     shape=(self.nORs, self.nODs),
            #     observed= self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
            # )
            #

            self.LogObservedDeaths = pm.Normal(
                "LogObservedDeaths",
                mu=pm.math.log(self.ExpectedDeaths[:, self.ObservedDaysIndx]),
                sigma=self.DeathNoiseScale,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            #
            self.Det("Observed", pm.math.exp(self.LogObservedDeaths), plot_trace=False)
            self.Det(
                "Z2",
                self.LogObservedDeaths - pm.math.log(self.ExpectedDeaths[:, self.CMDelayCut:]),
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            ax = plt.gca()
            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            nS, nD = self.trace.ExpectedDeaths[:, country_indx, :].shape

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                self.trace.ExpectedDeaths[:, country_indx, :] * np.exp(
                    np.repeat(self.trace.DeathsNoise.reshape((nS, 1)), nD, axis=1) * np.random.normal(
                        size=(self.trace.Infected[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            lines, labels = ax4.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                # ax2.legend(lines + lines2, labels + labels2, prop={"size": 8})
                ax4.legend(lines, labels, prop={"size": 8})


class CMActive(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array([0.00000000e+00, 2.30393104e-26, 1.38645939e-18, 2.27937595e-13,
                                   1.12748908e-09, 5.48448096e-07, 5.12994740e-05, 1.39554488e-03,
                                   1.44850037e-02, 6.92010991e-02, 1.74319084e-01, 2.56507864e-01,
                                   2.38765082e-01, 1.49790698e-01, 6.66717254e-02, 2.19611811e-02,
                                   5.54424146e-03, 1.10475846e-03, 1.78130069e-04, 2.37394851e-05])

        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.3

        self.CMDelayCut = 10

        self.ObservedDaysIndx = np.arange(10, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperGrowthRateMean_log = pm.HalfStudentT(
                "HyperGrowthRateMean_log", nu=10, sigma=0.3
            )
            self.HyperGrowthRateVar = pm.HalfStudentT(
                "HyperGrowthRateVar", nu=10, sigma=0.3
            )

            self.RegionGrowthRate_log = pm.Normal("RegionGrowthRate_log", self.HyperGrowthRateMean_log,
                                                  self.HyperGrowthRateVar,
                                                  shape=(self.nORs,))

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
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("DeathInfected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # use the theano convolution function, reshaping as required
            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            self.ExpectedConfirmed = pm.Deterministic("ExpectedConfirmed", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.ObservedConfirmedLog = pm.Normal(
                "ObservedConfirmedLog",
                pm.math.log(self.ExpectedConfirmed[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.Active[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedConfirmedLog - np.log(self.ExpectedConfirmed[:, self.CMDelayCut:]),
                plot_trace=False
            )

    def plot_region_predictions(self, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):
            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            ax = plt.gca()
            means_d, _, _, err_d = produce_CIs(
                self.trace.Infected_d[:, country_indx, :]
            )

            means_a, _, _, err_a = produce_CIs(
                self.trace.Infected_a[:, country_indx, :]
            )

            means_expected_active, _, _, err_expected_active = produce_CIs(
                self.trace.ExpectedActive[:, country_indx, :]
            )

            means_expected_deaths, _, _, err_expected_deaths = produce_CIs(
                self.trace.ExpectedDeaths[:, country_indx, :]
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]
            active = self.d.Active[country_indx, :]

            if self.nHODs > 0:
                means_ho, li_ho, ui_ho, err_ho = produce_CIs(
                    self.trace.HeldoutDaysObserved[:, country_indx, :]
                )

                plt.errorbar(
                    self.HeldoutDaysIndx,
                    means_ho,
                    yerr=err_ho,
                    fmt="-^",
                    linewidth=1,
                    markersize=2,
                    label="Heldout Pred Confirmed",
                    zorder=1,
                )
                plt.scatter(
                    self.HeldoutDaysIndx,
                    labels[self.HeldoutDaysIndx],
                    label="Heldout Confirmed",
                    marker="*",
                    color="tab:red",
                    zorder=3,
                )

            plt.errorbar(
                days_x,
                means_a,
                yerr=err_a,
                fmt="-D",
                linewidth=1,
                markersize=2,
                label="Infected Active",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.errorbar(
                days_x,
                means_d,
                yerr=err_d,
                fmt="-s",
                linewidth=1,
                markersize=2,
                label="Infected Deaths",
                zorder=1,
                color="tab:red",
                alpha=0.25,
            )

            plt.errorbar(
                days_x,
                means_expected_active,
                yerr=err_expected_active,
                fmt="-o",
                linewidth=1,
                markersize=2,
                label="Expected Active",
                zorder=2,
                color="tab:blue"
            )

            plt.errorbar(
                days_x,
                means_expected_deaths,
                yerr=err_expected_deaths,
                fmt="-o",
                linewidth=1,
                markersize=2,
                label="Expected Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Observed Dead",
                marker="o",
                s=8,
                color="tab:orange",
                alpha=0.75,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Observed Active",
                marker="o",
                s=6,
                color="tab:green",
                zorder=3,
                alpha=0.75
            )

            # plot countermeasures
            colors = ["tab:purple",
                      "tab:blue",
                      "silver",
                      "gray",
                      "black",
                      "tomato",
                      "tab:red",
                      "hotpink",
                      "tab:green"]

            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])[0].tolist()
                for c in changes:
                    height += 1
                    if CM_changes[cm, c] == 1:
                        plt.plot(
                            [c, c],
                            [0, 10 ** 7],
                            color="lightgrey",
                            linewidth=1,
                            zorder=-2,
                        )
                        plt.text(
                            (c - min_x) / (max_x - min_x),
                            1 - (0.035 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            transform=ax.transAxes,
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )
                    else:
                        plt.plot(
                            [c, c],
                            [0, 10 ** 7],
                            color="lightgrey",
                            linewidth=1,
                            zorder=-2,
                        )
                        plt.text(
                            (c - min_x) / (max_x - min_x),
                            1 - (0.035 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            transform=ax.transAxes,
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 7])

            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax1 = plt.gca()
            means_growth, _, _, err = produce_CIs(
                self.trace.ExpectedGrowth[:, country_indx, :]
            )
            actual_growth, _, _, err_act = produce_CIs(
                self.trace.Growth[:, country_indx, :]
            )

            plt.errorbar(
                days_x,
                np.exp(actual_growth),
                yerr=err_act,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Predicted Growth",
                zorder=1,
                color="tab:orange",
            )
            plt.errorbar(
                days_x,
                np.exp(means_growth),
                yerr=err,
                fmt="-D",
                linewidth=1,
                markersize=2,
                label="Expected Growth",
                zorder=2,
                color="tab:blue",
            )

            CMs = self.d.ActiveCMs[country_indx, :, :]
            nCMs, _ = CMs.shape
            CM_changes = CMs[:, 1:] - CMs[:, :-1]
            height = 0
            for cm in range(nCMs):
                changes = np.nonzero(CM_changes[cm, :])[0].tolist()
                for c in changes:
                    height += 1
                    if CM_changes[cm, c] == 1:
                        plt.plot(
                            [c, c], [0, 2], color="lightgrey", alpha=1, linewidth=1, zorder=-2
                        )
                        plt.text(
                            c,
                            2 - (0.05 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )
                    else:
                        plt.plot(
                            [c, c], [0, 2], "--r", alpha=0.5, linewidth=1, zorder=-2
                        )
                        plt.text(
                            c,
                            2 - (0.05 * (height)),
                            f"{cm + 1}",
                            color=colors[cm],
                            fontsize=5,
                            backgroundcolor="white",
                            horizontalalignment="center",
                            zorder=-1,
                            bbox=dict(
                                facecolor="white", edgecolor=colors[cm], boxstyle="round"
                            ),
                        )
            plt.ylim([0.7, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax2 = plt.gca()
            z1_mean, _, _, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, _, _, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            z3_mean, _, _, err_3 = produce_CIs(self.trace.Z3[:, country_indx, :])
            ln1 = plt.errorbar(
                days_x,
                z1_mean,
                yerr=err_1,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Growth Noise",
                zorder=1,
                color="tab:blue",
            )

            ln2 = plt.errorbar(
                self.ObservedDaysIndx,
                z3_mean,
                yerr=err_3,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Confirmation Noise",
                zorder=1,
                color="tab:green",
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-axis_scale * (max((np.max(err_3) + np.max(z3_mean)), (np.max(err_1) + np.max(z1_mean)))),
                      axis_scale * ((max((np.max(err_3) + np.max(z3_mean)), (np.max(err_1) + np.max(z1_mean)))))])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("Growth / Confirmation Noise")
            ax3 = plt.twinx()
            ln3 = plt.errorbar(
                self.ObservedDaysIndx,
                z2_mean,
                yerr=err_2,
                fmt="-x",
                linewidth=1,
                markersize=2,
                label="Death Noise",
                zorder=1,
                color="tab:orange",
            )
            plt.ylabel("Death Noise")

            plt.ylim(
                [-axis_scale * np.max(err_2 + np.max(z2_mean)), axis_scale * (np.max(err_2) + np.max(z2_mean))])
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx % 5 == 0:
                ax.legend(prop={"size": 8})
                ax1.legend(prop={"size": 8})
                ax2.legend(lines + lines2, labels + labels2, prop={"size": 8})

    def plot_effect(self, save_fig=True, output_dir="./out", x_min=0.5, x_max=1.5):
        assert self.trace is not None
        fig = plt.figure(figsize=(7, 3), dpi=300)
        means = np.mean(np.exp(-self.trace["CM_Alpha"]), axis=0)
        li = np.percentile(np.exp(-self.trace["CM_Alpha"]), 2.5, axis=0)
        ui = np.percentile(np.exp(-self.trace["CM_Alpha"]), 97.5, axis=0)
        lq = np.percentile(np.exp(-self.trace["CM_Alpha"]), 25, axis=0)
        uq = np.percentile(np.exp(-self.trace["CM_Alpha"]), 75, axis=0)

        N_cms = means.size

        plt.subplot(121)
        plt.plot([1, 1], [1, -(N_cms)], "--r", linewidth=0.5)
        y_vals = -1 * np.arange(N_cms)
        plt.scatter(means, y_vals, marker="|", color="k")
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.5)

        plt.xlim([x_min, x_max])
        plt.ylim([-(N_cms - 0.5), 0.5])
        plt.yticks(y_vals, self.d.CMs)
        plt.xlabel("Countermeasure Effectiveness")

        plt.subplot(122)
        correlation = np.corrcoef(np.exp(-self.trace["CM_Alpha"]), rowvar=False)
        plt.imshow(correlation, cmap="PuOr", vmin=-1, vmax=1)
        plt.colorbar()
        plt.yticks(np.arange(N_cms), [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.xticks(np.arange(N_cms), [f"$\\alpha_{{{i + 1}}}$" for i in range(N_cms)])
        plt.title("Correlation")

        plt.tight_layout()
        sns.despine()
        if save_fig:
            save_fig_pdf(output_dir, f"CMEffect")


class CMDeath_NB(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 10

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperGrowthRateMean_log = pm.HalfStudentT(
                "HyperGrowthRateMean_log", nu=10, sigma=0.3
            )
            self.HyperGrowthRateVar = pm.HalfStudentT(
                "HyperGrowthRateVar", nu=10, sigma=0.3
            )

            self.RegionGrowthRate_log = pm.Normal("RegionGrowthRate_log", self.HyperGrowthRateMean_log,
                                                  self.HyperGrowthRateVar,
                                                  shape=(self.nORs,))

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

            self.DailyGrowthNoise = 0.1

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", -3, 10, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # use the theano convolution function, reshaping as required
            # 1.034649e+00s
            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            # expected_confirmed = convolution(self.Infected, self.DelayProb, axis=1)
            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = pm.HalfNormal("Phi", 5)

            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
                alpha=self.Phi,
                shape=(self.nORs, self.nODs),
                observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
            )

            self.Det(
                "Z2",
                self.ObservedDeaths - self.ExpectedDeaths[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            ed_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ed_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMDeath_R(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 10
        self.DailyGrowthNoise = 0.1

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            def compute_R(growth_rate, alpha=(1 / (0.62 ** 2)), beta=(1 / (6.5 * 0.62 ** 2))):
                def M(z):
                    if z > beta:
                        raise ValueError("invalid z")
                    else:
                        return (1 - (z / beta)) ** -alpha

                return 1 / M(-(growth_rate - 1))

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperR = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

        self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

        self.InitialSize_log = pm.Normal("InitialSize_log", -3, 10, shape=(self.nORs,))
        self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
            self.nORs, 1)) + self.Growth.cumsum(axis=1))

        self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

        # use the theano convolution function, reshaping as required
        # 1.034649e+00s
        expected_confirmed = T.nnet.conv2d(
            self.Infected.reshape((1, 1, self.nORs, self.nDs)),
            np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
            border_mode="full",

        )[:, :, :, :self.nDs]

        # expected_confirmed = convolution(self.Infected, self.DelayProb, axis=1)

        self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_confirmed.reshape(
            (self.nORs, self.nDs)))

        # self.ObservedDeaths = pm.Normal(
        #     "ObservedDeaths",
        #     mu = self.ExpectedDeaths[:, self.ObservedDaysIndx],
        #     sigma = 0.25 * self.ExpectedDeaths[:, self.ObservedDaysIndx] + 0.25,
        #     shape=(self.nORs, self.nODs),
        #     observed= self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
        # )
        #

        self.LogObservedDeaths = pm.Normal(
            "LogObservedDeaths",
            mu=pm.math.log(self.ExpectedDeaths[:, self.ObservedDaysIndx]),
            sigma=self.DeathNoiseScale,
            shape=(self.nORs, self.nODs),
            observed=np.log(self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx])
        )

        #
        self.Det("Observed", pm.math.exp(self.LogObservedDeaths), plot_trace=False)
        self.Det(
            "Z2",
            self.LogObservedDeaths - pm.math.log(self.ExpectedDeaths[:, self.CMDelayCut:]),
            plot_trace=False
        )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            ax = plt.gca()
            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            nS, nD = self.trace.ExpectedDeaths[:, country_indx, :].shape

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                self.trace.ExpectedDeaths[:, country_indx, :] * np.exp(
                    np.repeat(self.trace.DeathsNoise.reshape((nS, 1)), nD, axis=1) * np.random.normal(
                        size=(self.trace.Infected[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            lines, labels = ax4.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                # ax2.legend(lines + lines2, labels + labels2, prop={"size": 8})
                ax4.legend(lines, labels, prop={"size": 8})


class CMActive_R(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.3

        self.CMDelayCut = 10

        self.Alpha = (1 / (0.62 ** 2))
        self.Beta = (1 / (6.5 * (0.62 ** 2)))

        self.ObservedDaysIndx = np.arange(10, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            def compute_R(growth_rate, alpha=(1 / (0.62 ** 2)), beta=(1 / (6.5 * 0.62 ** 2))):
                def M(z):
                    if z > beta:
                        raise ValueError("invalid z")
                    else:
                        return (1 - (z / beta)) ** -alpha

                return 1 / M(-(growth_rate - 1))

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperR = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               self.Beta * (T.power(pm.math.exp(self.ExpectedLogR),
                                                                    1 / self.Alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 10, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # use the theano convolution function, reshaping as required
            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            self.ExpectedConfirmed = pm.Deterministic("ExpectedConfirmed", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.ObservedConfirmedLog = pm.Normal(
                "ObservedConfirmedLog",
                pm.math.log(self.ExpectedConfirmed[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.Active[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedConfirmedLog - np.log(self.ExpectedConfirmed[:, self.CMDelayCut:]),
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            means_expected_active, lu_ea, up_ea, err_expected_active = produce_CIs(
                self.trace.ExpectedConfirmed[:, country_indx, :] * np.exp(
                    self.ConfirmationNoise * np.random.normal(
                        size=(self.trace.ExpectedConfirmed[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            active = self.d.Active[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_active,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Recorded Active Cases",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.plot([min_x, max_x], [1.0, 1.0], "--", color="lightgrey", linewidth=1)
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMDeath_R_NB(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

        self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

        self.InitialSize_log = pm.Normal("InitialSize_log", -6, 100, shape=(self.nORs,))
        self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
            self.nORs, 1)) + self.Growth.cumsum(axis=1))

        self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

        expected_confirmed = T.nnet.conv2d(
            self.Infected.reshape((1, 1, self.nORs, self.nDs)),
            np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
            border_mode="full"
        )[:, :, :, :self.nDs]

        self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_confirmed.reshape(
            (self.nORs, self.nDs)))

        self.Phi = pm.HalfNormal("Phi", 5)

        self.ObservedDeaths = pm.NegativeBinomial(
            "ObservedDeaths",
            mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
            alpha=6,
            shape=(self.nORs, self.nODs),
            observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
        )

        self.Det(
            "Z2",
            self.ExpectedDeaths[:, self.CMDelayCut:] - self.ObservedDeaths,
            plot_trace=False
        )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            # dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=6)
            ed_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ed_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMActive_R_NewCases_NB(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 10
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(3),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (T.power(pm.math.exp(self.ExpectedLogR), 1 / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

        self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

        self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
        self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
            self.nORs, 1)) + self.Growth.cumsum(axis=1))

        self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

        expected_confirmed = T.nnet.conv2d(
            self.Infected.reshape((1, 1, self.nORs, self.nDs)),
            np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
            border_mode="full",

        )[:, :, :, :self.nDs]

        self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
            (self.nORs, self.nDs)))

        self.Phi = 3.5

        self.ObservedCases = pm.NegativeBinomial(
            "ObservedCases",
            mu=self.ExpectedCases[:, self.ObservedDaysIndx],
            alpha=self.Phi,
            shape=(self.nORs, self.nODs),
            observed=self.d.NewCases[self.OR_indxs, :][:, self.ObservedDaysIndx]
        )

        self.Det(
            "Z2",
            self.ObservedCases - self.ExpectedCases[:, self.CMDelayCut:],
            plot_trace=False
        )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            # dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=self.Phi)
            ec_output = dist.random()

            means_cc, lu_cc, up_cc, err_cc = produce_CIs(
                ec_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Daily Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_cc,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_cc, up_cc, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -10, 10 ** 6])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMCombined_R(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProbActive = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                         0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                         0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                         0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                         0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                         0.00645582, 0.00534967, 0.00442695])

        self.DelayProbDeath = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.3

        self.CMDelayCut = 10

        self.Alpha = (1 / (0.62 ** 2))
        self.Beta = (1 / (6.5 * (0.62 ** 2)))

        self.ObservedDaysIndx = np.arange(10, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperR = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               self.Beta * (T.power(pm.math.exp(self.ExpectedLogR),
                                                                    1 / self.Alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(
                                                   self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "GrowthActive",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Normal(
                "GrowthDeath",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Det("Z1A", self.GrowthActive - self.ExpectedGrowth, plot_trace=False)
            self.Det("Z1D", self.GrowthDeath - self.ExpectedGrowth, plot_trace=False)

            # active cases
            self.InitialSizeActive_log = pm.Normal("InitialSizeActive_log", 1, 10, shape=(self.nORs,))
            self.InfectedActive_log = pm.Deterministic("InfectedActive_log", T.reshape(self.InitialSizeActive_log, (
                self.nORs, 1)) + self.GrowthActive.cumsum(axis=1))

            self.InfectedActive = pm.Deterministic("InfectedActive", pm.math.exp(self.InfectedActive_log))

            # use the theano convolution function, reshaping as required
            expected_active = T.nnet.conv2d(
                self.InfectedActive.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProbActive, newshape=(1, 1, 1, self.DelayProbActive.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            self.ExpectedActive = pm.Deterministic("ExpectedActive", expected_active.reshape(
                (self.nORs, self.nDs)))

            self.ObservedActiveLog = pm.Normal(
                "ObservedActiveLog",
                pm.math.log(self.ExpectedActive[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.Active[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedActiveLog - np.log(self.ExpectedActive[:, self.CMDelayCut:]),
                plot_trace=False
            )

            ## deaths
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", -3, 10, shape=(self.nORs,))
            self.InfectedDeaths_log = pm.Deterministic("InfectedDeaths_log", T.reshape(self.InitialSizeDeaths_log, (
                self.nORs, 1)) + self.GrowthDeath.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(self.InfectedDeaths_log))

            expected_deaths = T.nnet.conv2d(
                self.InfectedDeaths.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProbDeath, newshape=(1, 1, 1, self.DelayProbDeath.size)),
                border_mode="full",

            )[:, :, :, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # the higher, the less noisy! this fixed value is taken as the learnt death model value (a little lower, so more noisy)
            self.Phi = 3

            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
                alpha=self.Phi,
                shape=(self.nORs, self.nODs),
                observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
            )

            self.Det(
                "Z3",
                self.ObservedDeaths - self.ExpectedDeaths[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_a, lu_ia, up_ia, err_a = produce_CIs(
                self.trace.InfectedActive[:, country_indx, :]
            )

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.Phi]), nDs, axis=0).T)
            ed_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ed_output
            )

            means_expected_active, lu_ea, up_ea, err_expected_active = produce_CIs(
                self.trace.ExpectedActive[:, country_indx, :] * np.exp(
                    self.ConfirmationNoise * np.random.normal(
                        size=(self.trace.ExpectedActive[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]
            active = self.d.Active[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_a,
                label="Infected Active",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_ia, up_ia, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_active,
                label="Predicted Active Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Recorded Active Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx].data,
                label="Heldout Active Cases",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            plt.plot(
                days_x,
                means_d,
                label="Infected Deaths",
                zorder=1,
                color="tab:orange",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:orange", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            # part 2!
            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)
            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :]) - 1
            )

            actual_growth_active, lu_ag_active, up_ag_active, err_act_active = produce_CIs(
                np.exp(self.trace.GrowthActive[:, country_indx, :]) - 1
            )

            actual_growth_death, lu_ag_death, up_ag_death, err_act_death = produce_CIs(
                np.exp(self.trace.GrowthDeath[:, country_indx, :]) - 1
            )

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:gray")
            plt.plot(days_x, actual_growth_active, label="Predicted Active Cases Growth", zorder=1, color="tab:purple")
            plt.plot(days_x, actual_growth_death, label="Predicted Deaths Growth", zorder=1, color="tab:orange")
            plt.plot([min_x, max_x], [0, 0], "--", color="lightgrey")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:gray", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag_active, up_ag_active, alpha=0.25, color="tab:purple", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag_death, up_ag_death, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.ylim([-1, 1])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1A_mean, lu_z1A, up_z1A, err_1A = produce_CIs(self.trace.Z1A[:, country_indx, :])
            z1D_mean, lu_z1D, up_z1D, err_1D = produce_CIs(self.trace.Z1D[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            z3_mean, lu_z3, up_z3, err_3 = produce_CIs(self.trace.Z3[:, country_indx, :])

            plt.plot(days_x, z1A_mean, color="tab:purple", label="Active Cases Growth Noise")
            plt.fill_between(
                days_x, lu_z1A, up_z1A, alpha=0.25, color="tab:purple", linewidth=0
            )
            plt.plot(days_x, z1D_mean, color="tab:orange", label="Death Growth Noise")
            plt.fill_between(
                days_x, lu_z1D, up_z1D, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:blue", label="Active Cases Output Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("Growth Noise, Active Cases Output Noise")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z3_mean, color="tab:red", label="Death Output Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z3, up_z3, alpha=0.25, color="tab:red", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z3)), np.max(np.abs(lu_z3)))
            plt.ylabel("Death Output Noise")
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMCombined_R_NoiselessDeath(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProbActive = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                         0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                         0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                         0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                         0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                         0.00645582, 0.00534967, 0.00442695])

        self.DelayProbDeath = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.3

        self.CMDelayCut = 10

        self.Alpha = (1 / (0.62 ** 2))
        self.Beta = (1 / (6.5 * (0.62 ** 2)))

        self.ObservedDaysIndx = np.arange(10, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperR = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               self.Beta * (T.power(pm.math.exp(self.ExpectedLogR),
                                                                    1 / self.Alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(
                                                   self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "GrowthActive",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Det(
                "GrowthDeath",
                self.ExpectedGrowth,
                plot_trace=False
            )

            self.Det("Z1A", self.GrowthActive - self.ExpectedGrowth, plot_trace=False)
            self.Det("Z1D", self.GrowthDeath - self.ExpectedGrowth, plot_trace=False)

            # active cases
            self.InitialSizeActive_log = pm.Normal("InitialSizeActive_log", 1, 10, shape=(self.nORs,))
            self.InfectedActive_log = pm.Deterministic("InfectedActive_log", T.reshape(self.InitialSizeActive_log, (
                self.nORs, 1)) + self.GrowthActive.cumsum(axis=1))

            self.InfectedActive = pm.Deterministic("InfectedActive", pm.math.exp(self.InfectedActive_log))

            # use the theano convolution function, reshaping as required
            expected_active = T.nnet.conv2d(
                self.InfectedActive.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProbActive, newshape=(1, 1, 1, self.DelayProbActive.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            self.ExpectedActive = pm.Deterministic("ExpectedActive", expected_active.reshape(
                (self.nORs, self.nDs)))

            self.ObservedActiveLog = pm.Normal(
                "ObservedActiveLog",
                pm.math.log(self.ExpectedActive[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.Active[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedActiveLog - np.log(self.ExpectedActive[:, self.CMDelayCut:]),
                plot_trace=False
            )

            ## deaths
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", -3, 10, shape=(self.nORs,))
            self.InfectedDeaths_log = pm.Deterministic("InfectedDeaths_log", T.reshape(self.InitialSizeDeaths_log, (
                self.nORs, 1)) + self.GrowthDeath.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(self.InfectedDeaths_log))

            expected_deaths = T.nnet.conv2d(
                self.InfectedDeaths.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProbDeath, newshape=(1, 1, 1, self.DelayProbDeath.size)),
                border_mode="full",

            )[:, :, :, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # the higher, the less noisy! this fixed value is taken as the learnt death model value (a little lower, so more noisy)
            self.Phi = 3

            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
                alpha=self.Phi,
                shape=(self.nORs, self.nODs),
                observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
            )

            self.Det(
                "Z3",
                self.ObservedDeaths - self.ExpectedDeaths[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_a, lu_ia, up_ia, err_a = produce_CIs(
                self.trace.InfectedActive[:, country_indx, :]
            )

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.Phi]), nDs, axis=0).T)
            ed_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ed_output
            )

            means_expected_active, lu_ea, up_ea, err_expected_active = produce_CIs(
                self.trace.ExpectedActive[:, country_indx, :] * np.exp(
                    self.ConfirmationNoise * np.random.normal(
                        size=(self.trace.ExpectedActive[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]
            active = self.d.Active[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_a,
                label="Infected Active",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_ia, up_ia, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_active,
                label="Predicted Active Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Recorded Active Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx].data,
                label="Heldout Active Cases",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            plt.plot(
                days_x,
                means_d,
                label="Infected Deaths",
                zorder=1,
                color="tab:orange",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:orange", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 6])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            # part 2!
            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)
            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth_active, lu_ag_active, up_ag_active, err_act_active = produce_CIs(
                np.exp(self.trace.GrowthActive[:, country_indx, :])
            )

            median_actual_growth_active = np.percentile(np.exp(self.trace.GrowthActive[:, country_indx, :]), 50, axis=0)

            actual_growth_death, lu_ag_death, up_ag_death, err_act_death = produce_CIs(
                np.exp(self.trace.GrowthDeath[:, country_indx, :])
            )

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:gray")
            plt.plot(days_x, actual_growth_active, label="Predicted Active Cases Growth", zorder=1, color="tab:purple")
            plt.plot(days_x, actual_growth_death, label="Predicted Deaths Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, median_actual_growth_active, "--", label="Med Death Growth", zorder=1, color="tab:purple")
            plt.plot([min_x, max_x], [1, 1], "--", color="lightgrey", linewidth=0.5)

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:gray", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag_active, up_ag_active, alpha=0.25, color="tab:purple", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag_death, up_ag_death, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1A_mean, lu_z1A, up_z1A, err_1A = produce_CIs(self.trace.Z1A[:, country_indx, :])
            z1D_mean, lu_z1D, up_z1D, err_1D = produce_CIs(self.trace.Z1D[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            z3_mean, lu_z3, up_z3, err_3 = produce_CIs(self.trace.Z3[:, country_indx, :])

            plt.plot(days_x, z1A_mean, color="tab:purple", label="Active Cases Growth Noise")
            plt.fill_between(
                days_x, lu_z1A, up_z1A, alpha=0.25, color="tab:purple", linewidth=0
            )
            plt.plot(days_x, z1D_mean, color="tab:orange", label="Death Growth Noise")
            plt.fill_between(
                days_x, lu_z1D, up_z1D, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:blue", label="Active Cases Output Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("Growth Noise, Active Cases Output Noise")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z3_mean, color="tab:red", label="Death Output Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z3, up_z3, alpha=0.25, color="tab:red", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z3)), np.max(np.abs(lu_z3)))
            plt.ylabel("Death Output Noise")
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMActive_R_NewCases(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.3

        self.CMDelayCut = 10

        self.Alpha = (1 / (0.62 ** 2))
        self.Beta = (1 / (6.5 * (0.62 ** 2)))

        self.ObservedDaysIndx = np.arange(10, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperR = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2.5),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               self.Beta * (T.power(pm.math.exp(self.ExpectedLogR),
                                                                    1 / self.Alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # use the theano convolution function, reshaping as required
            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            self.ExpectedConfirmed = pm.Deterministic("ExpectedConfirmed", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.ObservedConfirmedLog = pm.Normal(
                "ObservedConfirmedLog",
                pm.math.log(self.ExpectedConfirmed[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.NewCases[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedConfirmedLog - np.log(self.ExpectedConfirmed[:, self.CMDelayCut:]),
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            means_expected_active, lu_ea, up_ea, err_expected_active = produce_CIs(
                self.trace.ExpectedConfirmed[:, country_indx, :] * np.exp(
                    self.ConfirmationNoise * np.random.normal(
                        size=(self.trace.ExpectedConfirmed[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            active = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_active,
                label="Predicted Deaths",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx].data,
                label="Heldout New Cases",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.plot([min_x, max_x], [1.0, 1.0], "--", color="lightgrey", linewidth=1)
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMCombined_R_NewCases(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProbActive = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                         0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                         0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                         0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                         0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                         0.00645582, 0.00534967, 0.00442695])

        self.DelayProbDeath = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.DailyGrowthNoise = 0.1
        self.ConfirmationNoise = 0.3

        self.CMDelayCut = 10

        self.Alpha = (1 / (0.62 ** 2))
        self.Beta = (1 / (6.5 * (0.62 ** 2)))

        self.ObservedDaysIndx = np.arange(10, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperR = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(3),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               self.Beta * (pm.math.exp(self.ExpectedLogR / self.Alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(
                                                   self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "GrowthActive",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Normal(
                "GrowthDeath",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

            self.Det("Z1A", self.GrowthActive - self.ExpectedGrowth, plot_trace=False)
            self.Det("Z1D", self.GrowthDeath - self.ExpectedGrowth, plot_trace=False)

            # active cases
            self.InitialSizeActive_log = pm.Normal("InitialSizeActive_log", 1, 10, shape=(self.nORs,))
            self.InfectedActive_log = pm.Deterministic("InfectedActive_log", T.reshape(self.InitialSizeActive_log, (
                self.nORs, 1)) + self.GrowthActive.cumsum(axis=1))

            self.InfectedActive = pm.Deterministic("InfectedActive", pm.math.exp(self.InfectedActive_log))

            # use the theano convolution function, reshaping as required
            expected_active = T.nnet.conv2d(
                self.InfectedActive.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProbActive, newshape=(1, 1, 1, self.DelayProbActive.size)),
                border_mode="full",
            )[:, :, :, :self.nDs]

            self.ExpectedActive = pm.Deterministic("ExpectedActive", expected_active.reshape(
                (self.nORs, self.nDs)))

            self.ObservedActiveLog = pm.Normal(
                "ObservedActiveLog",
                pm.math.log(self.ExpectedActive[:, self.ObservedDaysIndx]),
                self.ConfirmationNoise,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.NewCases[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            # self.Det("Observed", pm.math.exp(self.Observed_log), plot_trace=False)
            self.Det(
                "Z2",
                self.ObservedActiveLog - np.log(self.ExpectedActive[:, self.CMDelayCut:]),
                plot_trace=False
            )

            ## deaths
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", -3, 10, shape=(self.nORs,))
            self.InfectedDeaths_log = pm.Deterministic("InfectedDeaths_log", T.reshape(self.InitialSizeDeaths_log, (
                self.nORs, 1)) + self.GrowthDeath.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(self.InfectedDeaths_log))

            expected_deaths = T.nnet.conv2d(
                self.InfectedDeaths.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProbDeath, newshape=(1, 1, 1, self.DelayProbDeath.size)),
                border_mode="full",

            )[:, :, :, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # the higher, the less noisy! this fixed value is taken as the learnt death model value (a little lower, so more noisy)
            self.Phi = 4

            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
                alpha=self.Phi,
                shape=(self.nORs, self.nODs),
                observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
            )

            self.Det(
                "Z3",
                self.ObservedDeaths - self.ExpectedDeaths[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_a, lu_ia, up_ia, err_a = produce_CIs(
                self.trace.InfectedActive[:, country_indx, :]
            )

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.Phi]), nDs, axis=0).T)
            ed_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ed_output
            )

            means_expected_active, lu_ea, up_ea, err_expected_active = produce_CIs(
                self.trace.ExpectedActive[:, country_indx, :] * np.exp(
                    self.ConfirmationNoise * np.random.normal(
                        size=(self.trace.ExpectedActive[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]
            active = self.d.Active[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_a,
                label="Infected Active",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_ia, up_ia, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_active,
                label="Predicted Active Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx],
                label="Recorded Active Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                active[self.ObservedDaysIndx].data,
                label="Heldout Active Cases",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            plt.plot(
                days_x,
                means_d,
                label="Infected Deaths",
                zorder=1,
                color="tab:orange",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:orange", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 6])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            # part 2!
            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)
            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth_active, lu_ag_active, up_ag_active, err_act_active = produce_CIs(
                np.exp(self.trace.GrowthActive[:, country_indx, :])
            )

            median_actual_growth_active = np.percentile(np.exp(self.trace.GrowthActive[:, country_indx, :]), 50, axis=0)

            actual_growth_death, lu_ag_death, up_ag_death, err_act_death = produce_CIs(
                np.exp(self.trace.GrowthDeath[:, country_indx, :])
            )

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:gray")
            plt.plot(days_x, actual_growth_active, label="Predicted Active Cases Growth", zorder=1, color="tab:purple")
            plt.plot(days_x, actual_growth_death, label="Predicted Deaths Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, median_actual_growth_active, "--", label="Med Death Growth", zorder=1, color="tab:purple")
            plt.plot([min_x, max_x], [1, 1], "--", color="lightgrey", linewidth=0.5)

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:gray", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag_active, up_ag_active, alpha=0.25, color="tab:purple", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag_death, up_ag_death, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1A_mean, lu_z1A, up_z1A, err_1A = produce_CIs(self.trace.Z1A[:, country_indx, :])
            z1D_mean, lu_z1D, up_z1D, err_1D = produce_CIs(self.trace.Z1D[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            z3_mean, lu_z3, up_z3, err_3 = produce_CIs(self.trace.Z3[:, country_indx, :])

            plt.plot(days_x, z1A_mean, color="tab:purple", label="Active Cases Growth Noise")
            plt.fill_between(
                days_x, lu_z1A, up_z1A, alpha=0.25, color="tab:purple", linewidth=0
            )
            plt.plot(days_x, z1D_mean, color="tab:orange", label="Death Growth Noise")
            plt.fill_between(
                days_x, lu_z1D, up_z1D, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:blue", label="Active Cases Output Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("Growth Noise, Active Cases Output Noise")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z3_mean, color="tab:red", label="Death Output Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z3, up_z3, alpha=0.25, color="tab:red", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z3)), np.max(np.abs(lu_z3)))
            plt.ylabel("Death Output Noise")
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class death_test_r_nb(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 10
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            # self.HyperRMean = pm.StudentT(
            #     "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            # )
            # self.HyperRVar = pm.HalfStudentT(
            #     "HyperRVar", nu=10, sigma=0.3
            # )

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3),
                                        0.2,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

        self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

        self.InitialSize_log = pm.Normal("InitialSize_log", -6, 100, shape=(self.nORs,))
        self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
            self.nORs, 1)) + self.Growth.cumsum(axis=1))

        self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

        expected_confirmed = T.nnet.conv2d(
            self.Infected.reshape((1, 1, self.nORs, self.nDs)),
            np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
            border_mode="full"
        )[:, :, :, :self.nDs]

        self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_confirmed.reshape(
            (self.nORs, self.nDs)))

        self.Phi = pm.HalfNormal("Phi", 5)

        self.ObservedDeaths = pm.NegativeBinomial(
            "ObservedDeaths",
            mu=self.ExpectedDeaths[:, self.ObservedDaysIndx],
            alpha=self.Phi,
            shape=(self.nORs, self.nODs),
            observed=self.d.NewDeaths[self.OR_indxs, :][:, self.ObservedDaysIndx]
        )

        self.Det(
            "Z2",
            self.ExpectedDeaths[:, self.CMDelayCut:] - self.ObservedDeaths,
            plot_trace=False
        )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            ed_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ed_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -1, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class active_test_r_nb(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 10
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3),
                                        0.2,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",

            )[:, :, :, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = 3.5

            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases[:, self.ObservedDaysIndx],
                alpha=self.Phi,
                shape=(self.nORs, self.nODs),
                observed=self.d.NewCases[self.OR_indxs, :][:, self.ObservedDaysIndx]
            )

            self.Det(
                "Z2",
                self.ObservedCases - self.ExpectedCases[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            # dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=self.Phi)
            ec_output = dist.random()

            means_cc, lu_cc, up_cc, err_cc = produce_CIs(
                ec_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Daily Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_cc,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_cc, up_cc, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -10, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class active_test_r_ln(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 10
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3),
                                        0.2,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            expected_confirmed = T.nnet.conv2d(
                self.Infected.reshape((1, 1, self.nORs, self.nDs)),
                np.reshape(self.DelayProb, newshape=(1, 1, 1, self.DelayProb.size)),
                border_mode="full",

            )[:, :, :, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = 3.5

            self.ObservedCases = pm.Normal(
                "ObservedCases",
                mu=pm.math.log(self.ExpectedCases[:, self.ObservedDaysIndx]),
                sigma=0.3,
                shape=(self.nORs, self.nODs),
                observed=np.log(self.d.NewCases[self.OR_indxs, :][:, self.ObservedDaysIndx])
            )

            self.Det(
                "Z2",
                self.ObservedCases - self.ExpectedCases[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            # dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=self.Phi)
            ec_output = dist.random()

            means_cc, lu_cc, up_cc, err_cc = produce_CIs(
                ec_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 5
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Daily Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_cc,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_cc, up_cc, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -10, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            ax4.twinx()
            ax5 = plt.gca()
            plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )
            y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMDeath_Final(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.DelayProb = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

        observed = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                if self.d.NewDeaths[r, d] > 0:
                    observed.append(r * self.nDs + d)
        self.observed_days = np.array(observed)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # growth model
            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(2),
            )
            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", self.HyperRMean,
                                        self.HyperRVar,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,

            )

        self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

        self.InitialSize_log = pm.Normal("InitialSize_log", -6, 100, shape=(self.nORs,))
        self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
            self.nORs, 1)) + self.Growth.cumsum(axis=1))

        self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

        expected_confirmed = C.conv2d(
            self.Infected,
            np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
            border_mode="full"
        )[:, :self.nDs]

        self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_confirmed.reshape(
            (self.nORs, self.nDs)))

        self.LogObservedDeaths = pm.Normal(
            "LogObservedDeaths",
            mu=pm.math.log(self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.observed_days]),
            sigma=0.4,
            shape=(len(self.observed_days),),
            observed=np.log(self.d.NewDeaths.reshape((self.nORs * self.nDs,))[self.observed_days])
        )

        # self.Z2 = pm.Deterministic("Z2",
        #     self.LogObservedDeaths - np.log(self.d.NewDeaths.reshape((self.nORs * self.nDs, ))[self.observed_days])
        # )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                self.trace.ExpectedDeaths[:, country_indx, :] * np.exp(
                    0.4 * np.random.normal(
                        size=(self.trace.ExpectedDeaths[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 25
            max_x = len(days) - 1

            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Infected",
                zorder=1,
                color="tab:blue",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:blue", linewidth=0
            )

            plt.plot(
                days_x,
                means_expected_deaths,
                label="Predicted Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="Recorded New Deaths",
                marker="o",
                s=10,
                color="black",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 4])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            # ax4.twinx()
            # ax5 = plt.gca()
            # plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            # plt.fill_between(
            #     self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            # )
            # y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            # plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            # lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                # ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMActive_Final(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3),
                                        0.2,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            expected_confirmed = C.conv2d(
                self.Infected,
                np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.ObservedCases = pm.Normal(
                "ObservedCases",
                mu=pm.math.log(self.ExpectedCases[:, self.ObservedDaysIndx] + 1e-6) * self.d.NewCases.data > 0,
                sigma=0.3,
                shape=(self.nORs, self.nODs),
                observed=np.log(
                    self.d.NewCases[self.OR_indxs, :][:, self.ObservedDaysIndx] + 1e-6) * self.d.NewCases.data > 0
            )

            self.Det(
                "Z2",
                self.ObservedCases - self.ExpectedCases[:, self.CMDelayCut:],
                plot_trace=False
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            means_ea, lu_ea, up_ea, err_eea = produce_CIs(
                self.trace.ExpectedCases[:, country_indx, :] * np.exp(
                    0.3 * np.random.normal(
                        size=(self.trace.ExpectedCases[:, country_indx, :].shape)))
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 25
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Daily Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_ea,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            # ax4.twinx()
            # ax5 = plt.gca()
            # plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            # plt.fill_between(
            #     self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            # )
            # y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            # plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})


class CMActive_Final_NB(BaseCMModel):
    def __init__(
            self, data, output_model="lognorm", name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

        observed = []
        for r in range(self.nRs):
            skipped_days = []
            for d in range(self.nDs):
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(self.d.Confirmed.data[r, d]):
                    observed.append(r * self.nDs + d)
                else:
                    skipped_days.append(d)

            if len(skipped_days) > 0:
                print(f"Skipped day {[(data.Ds[sk].day, data.Ds[sk].month) for sk in skipped_days]} for {data.Rs[r]}")
        self.observed_days = np.array(observed)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3),
                                        0.2,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(
                                                   self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(
                                                   self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            expected_confirmed = C.conv2d(
                self.Infected,
                np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = pm.HalfNormal("Phi", 5)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.observed_days],
                alpha=self.Phi,
                shape=(len(self.observed_days),),
                observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.observed_days]
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            means_ea, lu_ea, up_ea, err_eea = produce_CIs(
                self.trace.ExpectedCases[:, country_indx, :] * np.exp(
                    0.3 * np.random.normal(
                        size=(self.trace.ExpectedCases[:, country_indx, :].shape)))
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            ec_output = dist.random()

            means_ea, lu_ea, up_ea, err_eea = produce_CIs(
                ec_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 25
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Daily Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_ea,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            # ax4.twinx()
            # ax5 = plt.gca()
            # plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            # plt.fill_between(
            #     self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            # )
            # y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            # plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            # lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                # ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})

class CMCombined_Final(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProbActive = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.DelayProbDeath = np.array(
            [
                0,
                2.10204045e-06,
                3.22312869e-05,
                1.84979560e-04,
                6.31412913e-04,
                1.53949439e-03,
                3.07378372e-03,
                5.32847235e-03,
                8.32057678e-03,
                1.19864352e-02,
                1.59626950e-02,
                2.02752812e-02,
                2.47013776e-02,
                2.90892369e-02,
                3.30827134e-02,
                3.66035310e-02,
                3.95327745e-02,
                4.19039762e-02,
                4.35677913e-02,
                4.45407357e-02,
                4.49607434e-02,
                4.47581467e-02,
                4.40800885e-02,
                4.28367817e-02,
                4.10649618e-02,
                3.93901360e-02,
                3.71499615e-02,
                3.48922699e-02,
                3.24149652e-02,
                3.00269472e-02,
                2.76836725e-02,
                2.52794388e-02,
                2.29349630e-02,
                2.07959867e-02,
                1.86809336e-02,
                1.67279378e-02,
                1.50166767e-02,
                1.33057159e-02,
                1.17490048e-02,
                1.03030011e-02,
                9.10633952e-03,
                7.97333972e-03,
                6.95565185e-03,
                6.05717970e-03,
                5.25950540e-03,
                4.61137626e-03,
                3.94442886e-03,
                3.37948046e-03,
                2.91402865e-03,
                2.48911619e-03,
                2.14007737e-03,
                1.81005702e-03,
                1.54339818e-03,
                1.32068199e-03,
                1.11358095e-03,
                9.53425490e-04,
                7.99876440e-04,
                6.76156345e-04,
                5.68752088e-04,
                4.93278826e-04,
                4.08596625e-04,
                3.37127249e-04,
                2.92283720e-04,
                2.41934846e-04,
                1.98392580e-04,
            ]
        )

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.15

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

        observed_active = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(self.d.Confirmed.data[r, d]):
                    observed_active.append(r * self.nDs + d)

        self.all_observed_active = np.array(observed_active)

        observed_deaths = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 10 deaths
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed_deaths.append(r * self.nDs + d)

        self.all_observed_active = np.array(observed_deaths)



    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3),
                                        0.2,
                                        shape=(self.nORs,))

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            alpha = (1 / (0.62 ** 2))
            beta = (1 / (6.5 * (0.62 ** 2)))

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           pm.math.log(
                                               beta * (pm.math.exp(
                                                   self.ExpectedLogR / alpha) - T.ones_like(
                                                   self.ExpectedLogR)) + T.ones_like(
                                                   self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Normal(
                "Growth",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", 1, 100, shape=(self.nORs,))
            self.Infected_log = pm.Deterministic("Infected_log", T.reshape(self.InitialSize_log, (
                self.nORs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            expected_confirmed = C.conv2d(
                self.Infected,
                np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = pm.HalfNormal("Phi", 5)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.observed_days],
                alpha=self.Phi,
                shape=(len(self.observed_days),),
                observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.observed_days]
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_d, lu_id, up_id, err_d = produce_CIs(
                self.trace.Infected[:, country_indx, :]
            )

            means_ea, lu_ea, up_ea, err_eea = produce_CIs(
                self.trace.ExpectedCases[:, country_indx, :] * np.exp(
                    0.3 * np.random.normal(
                        size=(self.trace.ExpectedCases[:, country_indx, :].shape)))
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            ec_output = dist.random()

            means_ea, lu_ea, up_ea, err_eea = produce_CIs(
                ec_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 25
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_d,
                label="Daily Infected",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_ea,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ea, up_ea, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="Recorded New Cases",
                marker="o",
                s=10,
                color="tab:green",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx].data,
                label="Heldout New Deaths",
                marker="o",
                s=12,
                edgecolor="tab:green",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.Growth[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.Growth[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected Growth", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted Growth", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0.5, 2])
            plt.xlim([min_x, max_x])
            plt.ylabel("Growth")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1_mean, color="tab:blue", label="Growth Noise")
            plt.fill_between(
                days_x, lu_z1, up_z1, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([-2, 2])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$Z$")

            # ax4.twinx()
            # ax5 = plt.gca()
            # plt.plot(self.ObservedDaysIndx, z2_mean, color="tab:orange", label="Death Noise")
            # plt.fill_between(
            #     self.ObservedDaysIndx, lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            # )
            # y_lim = max(np.max(np.abs(up_z2)), np.max(np.abs(lu_z2)))
            # plt.ylim([-1.5 * y_lim, 1.5 * y_lim])

            plt.xlim([min_x, max_x])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            lines, labels = ax4.get_legend_handles_labels()
            # lines2, labels2 = ax5.get_legend_handles_labels()

            sns.despine(ax=ax)
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            sns.despine(ax=ax3)

            if country_indx % 5 == 4 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}",
                    )

            elif country_indx == 0:
                ax.legend(prop={"size": 8}, loc="center left")
                ax2.legend(prop={"size": 8}, loc="lower left")
                # ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})