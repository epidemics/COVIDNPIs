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

from epimodel.pymc3_models.utils import shift_right

log = logging.getLogger(__name__)
sns.set_style("ticks")

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")


def save_fig_pdf(output_dir, figname):
    datetime_str = datetime.now().strftime("%d-%m;%H-%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(f"Saving Plot at {os.path.abspath(output_dir)} at {datetime_str}")
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
            self, data, name="", model=None
    ):
        super().__init__(name, model)
        self.d = data
        self.plot_trace_vars = set()
        self.trace = None
        self.heldout_day_labels = None

    def LN(self, name, mean, log_var, plot_trace=True, shape=None):
        """Create a lognorm variable, adding it to self as attribute."""
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
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
            self, data, name="", model=None
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
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Confirmed.data[r, d]):
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
                                           beta * (pm.math.exp(
                                               self.ExpectedLogR / alpha) - T.ones_like(
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
        self.DelayProbCases = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                        0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                        0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                        0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                        0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                        0.00645582, 0.00534967, 0.00442695])

        self.DelayProbDeaths = np.array(
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
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Confirmed.data[r, d]):
                    observed_active.append(r * self.nDs + d)

        self.all_observed_active = np.array(observed_active)

        observed_deaths = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 10 deaths
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed_deaths.append(r * self.nDs + d)

        self.all_observed_deaths = np.array(observed_deaths)

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
                "GrowthCases",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Normal(
                "GrowthDeaths",
                self.ExpectedGrowth,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1C", self.GrowthCases - self.ExpectedGrowth, plot_trace=False)
            self.Det("Z1D", self.GrowthDeaths - self.ExpectedGrowth, plot_trace=False)

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 1, 20, shape=(self.nORs,))
            self.InfectedCases_log = pm.Deterministic("InfectedCases_log", T.reshape(self.InitialSizeCases_log, (
                self.nORs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(self.InfectedCases_log))

            expected_cases = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbCases, newshape=(1, self.DelayProbCases.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nORs, self.nDs)))

            self.Phi_1 = 6

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                alpha=self.Phi_1,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
            )

            self.Z2C = pm.Deterministic(
                "Z2C",
                self.ObservedCases - self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", -3, 20, shape=(self.nORs,))
            self.InfectedDeaths_log = pm.Deterministic("InfectedDeaths_log", T.reshape(self.InitialSizeDeaths_log, (
                self.nORs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(self.InfectedDeaths_log))

            expected_deaths = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            self.Phi_2 = 5

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                alpha=self.Phi_2,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]
            )

            self.Det(
                "Z2D",
                self.ObservedDeaths - self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]
            )

    def plot_region_predictions(self, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for country_indx, region in zip(self.OR_indxs, self.ORs):

            if country_indx % 5 == 0:
                plt.figure(figsize=(12, 20), dpi=300)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 1)

            means_ic, lu_ic, up_ic, err_ic = produce_CIs(
                self.trace.InfectedCases[:, country_indx, :]
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)
            ec_output = dist.random()

            means_ec, lu_ec, up_ec, err_ec = produce_CIs(
                ec_output
            )

            means_id, lu_id, up_id, err_id = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            # dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_2]), nDs, axis=0).T)
            dist = pm.NegativeBinomial.dist(mu=ed, alpha=5)
            # ed_output = dist.random()

            means_ed, lu_ed, up_ed, err_ed = produce_CIs(
                ed
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 25
            max_x = len(days) - 1

            newcases = self.d.NewCases[country_indx, :]
            deaths = self.d.NewDeaths[country_indx, :]

            ax = plt.gca()
            plt.plot(
                days_x,
                means_ic,
                label="Daily Infected - Cases",
                zorder=1,
                color="tab:purple",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_ic, up_ic, alpha=0.15, color="tab:purple", linewidth=0
            )

            plt.plot(
                days_x,
                means_ec,
                label="Predicted New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ec, up_ec, alpha=0.25, color="tab:blue", linewidth=0
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
                label="Heldout New Cases",
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
                means_id,
                label="Daily Infected - Deaths",
                zorder=1,
                color="tab:orange",
                alpha=0.25
            )

            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.15, color="tab:orange", linewidth=0
            )

            plt.plot(
                days_x,
                means_ed,
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
                label="Recorded Deaths",
                marker="o",
                s=10,
                color="tab:gray",
                alpha=0.9,
                zorder=3,
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx].data,
                label="Recorded Heldout Deaths",
                marker="o",
                s=12,
                edgecolor="tab:gray",
                facecolor="white",
                linewidth=1,
                alpha=0.9,
                zorder=2,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** 0, 10 ** 6])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_g, lu_g, up_g, err_g = produce_CIs(
                np.exp(self.trace.ExpectedGrowth[:, country_indx, :])
            )

            means_agc, lu_agc, up_agc, err_agc = produce_CIs(
                np.exp(self.trace.GrowthCases[:, country_indx, :])
            )

            means_agd, lu_agd, up_agd, err_agd = produce_CIs(
                np.exp(self.trace.GrowthDeaths[:, country_indx, :])
            )

            med_agc = np.percentile(np.exp(self.trace.GrowthCases[:, country_indx, :]), 50, axis=0)
            med_agd = np.percentile(np.exp(self.trace.GrowthDeaths[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, means_g, label="Predicted Growth", zorder=1, color="tab:gray")
            plt.plot(days_x, means_agc, label="Corrupted Growth - Cases", zorder=1, color="tab:purple")
            # plt.plot(days_x, med_agc, "--", color="tab:purple")
            plt.plot(days_x, means_agd, label="Corrupted Growth - Deaths", zorder=1, color="tab:orange")
            # plt.plot(days_x, med_agd, "--", color="tab:orange")

            plt.fill_between(days_x, lu_g, up_g, alpha=0.25, color="tab:gray", linewidth=0)
            plt.fill_between(days_x, lu_agc, up_agc, alpha=0.25, color="tab:purple", linewidth=0)
            plt.fill_between(days_x, lu_agd, up_agd, alpha=0.25, color="tab:orange", linewidth=0)

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
            z1C_mean, lu_z1C, up_z1C, err_1C = produce_CIs(self.trace.Z1C[:, country_indx, :])
            z1D_mean, lu_z1D, up_z1D, err_1D = produce_CIs(self.trace.Z1D[:, country_indx, :])
            # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            plt.plot(days_x, z1C_mean, color="tab:purple", label="Growth Noise - Cases")
            plt.fill_between(
                days_x, lu_z1C, up_z1C, alpha=0.25, color="tab:purple", linewidth=0
            )
            plt.plot(days_x, z1D_mean, color="tab:purple", label="Growth Noise - Deaths")
            plt.fill_between(
                days_x, lu_z1D, up_z1D, alpha=0.25, color="tab:orange", linewidth=0
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


class CMActive_Final_ICL(BaseCMModel):
    def __init__(
            self, data, name="", model=None
    ):
        super().__init__(data, name=name, model=model)

        self.SI = np.array(
            [0.04656309, 0.08698277, 0.1121656, 0.11937737, 0.11456359,
             0.10308026, 0.08852893, 0.07356104, 0.059462, 0.04719909,
             0.03683025, 0.02846977, 0.02163222, 0.01640488, 0.01221928,
             0.00903811, 0.00670216, 0.00490314, 0.00361434, 0.00261552,
             0.00187336, 0.00137485, 0.00100352, 0.00071164, 0.00050852,
             0.00036433, 0.00025036]
        )

        self.SI_rev = self.SI[::-1]
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
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Confirmed.data[r, d]):
                    observed.append(r * self.nDs + d)
                else:
                    skipped_days.append(d)

            if len(skipped_days) > 0:
                # print(f"Skipped day {[(data.Ds[sk].day, data.Ds[sk].month) for sk in skipped_days]} for {data.Rs[r]}")
                pass

        self.observed_days = np.array(observed)

    def build_model(self):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, 0.2, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(3.5),
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
                    * self.ActiveCMs[self.OR_indxs, :, :]
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            self.Normal(
                "LogR",
                self.ExpectedLogR,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            self.Det("Z1", self.LogR - self.ExpectedLogR, plot_trace=False)

            self.HyperNMean = pm.StudentT(
                "HyperNMean", nu=10, sigma=1, mu=np.log(3.5),
            )
            self.HyperNVar = pm.HalfStudentT(
                "HyperNVar", nu=10, sigma=5
            )

            self.InitialSize_log = pm.Normal("InitialSize_log", self.HyperNMean, self.HyperNVar, shape=(self.nORs,))

            # conv padding
            conv_padding = self.SI.size

            infected = T.zeros((self.nORs, self.nDs + conv_padding))
            infected = T.set_subtensor(infected[:, :conv_padding],
                                       pm.math.exp(self.InitialSize_log).reshape((self.nORs, 1)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, d].reshape((self.nORs, 1)) * infected[:, d:d + conv_padding] * self.SI_rev.reshape(
                        (1, conv_padding)), axis=1)
                infected = T.set_subtensor(infected[:, d + conv_padding], val)

            self.Infected = pm.Deterministic("Infected", infected[:, conv_padding:])

            expected_confirmed = C.conv2d(
                self.Infected,
                np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = pm.HalfNormal("Phi", 5)

            self.NewCases = pm.Data("NewCases", self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.observed_days])

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.observed_days],
                alpha=self.Phi,
                shape=(len(self.observed_days),),
                observed= self.NewCases
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
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=15)
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
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.LogR[:, country_indx, :])
            )

            med_growth = np.percentile(np.exp(self.trace.ExpectedLogR[:, country_indx, :]), 50, axis=0)

            plt.plot(days_x, med_growth, "--", label="Median Growth",
                     color="tab:blue")

            plt.plot(days_x, means_growth, label="Expected R", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted R", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0, 5])
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
            nS, nDs = self.trace.Z1[:, country_indx, :].shape
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(
                np.repeat(self.d.NewCases[country_indx, :].data.reshape(1, nDs), nS, axis=0) - self.trace.ExpectedCases[
                                                                                               :, country_indx, :])

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
            plt.plot(np.arange(self.nDs), z2_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                np.arange(self.nDs), lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
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
