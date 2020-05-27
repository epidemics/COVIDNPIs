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
import matplotlib.patches as patches

fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")

# taken from Cereda et. al (2020).
# https://arxiv.org/ftp/arxiv/papers/2003/2003.09320.pdf
# alpha is shape, beta is inverse scale (reciprocal reported in the paper).
SI_ALPHA = 1.87
SI_BETA = 0.28


# ICL paper versions.
# SI_ALPHA = (1 / (0.62 ** 2))
# SI_BETA = (1 / (6.5 * (0.62 ** 2)))


def save_fig_pdf(output_dir, figname):
    datetime_str = datetime.now().strftime("%d-%m;%H-%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(f"Saving Plot at {os.path.abspath(output_dir)} at {datetime_str}")
    plt.savefig(f"{output_dir}/{figname}_t{datetime_str}.pdf", bbox_inches='tight')


def produce_CIs(data):
    means = np.median(data, axis=0)
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

            if c < min_x:
                c_p = min_x
            else:
                c_p = c

            if CM_changes[cm, c] == 1:
                plt.text(c_p, plot_height, plot_style[cm][0], fontproperties=fp2, color=plot_style[cm][1], size=8,
                         va='center', ha='center', clip_on=True, zorder=1)
            else:
                plt.text(c_p, plot_height, plot_style[cm][0], fontproperties=fp2, color=plot_style[cm][1], size=8,
                         va='center', ha='center', clip_on=True, zorder=1)
                plt.plot([c_p - 1.5, c + 1.5], [plot_height - 0.005, plot_height + 0.005], color="black", zorder=2)

    plt.yticks([])
    return ax2


class BaseCMModel(Model):
    def __init__(
            self, data, cm_plot_style, name="", model=None
    ):
        super().__init__(name, model)
        self.d = data
        self.plot_trace_vars = set()
        self.trace = None
        self.heldout_day_labels = None

        if cm_plot_style is not None:
            self.cm_plot_style = cm_plot_style
        else:
            self.cm_plot_style = [
                # ("\uf7f2", "tab:red"),  # hospital symbol
                ("\uf963", "black"),  # mask
                ("\uf492", "mediumblue"),  # vial
                ("\uf0c0", "lightgrey"),  # ppl
                ("\uf0c0", "grey"),  # ppl
                ("\uf0c0", "black"),  # ppl
                ("\uf07a", "tab:orange"),  # shop 1
                ("\uf07a", "tab:red"),  # shop2
                ("\uf19d", "black"),  # school
                ("\uf965", "black")  # home
            ]

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
        fig = plt.figure(figsize=(9, 3), dpi=300)
        plt.subplot(121)
        self.d.coactivation_plot(self.cm_plot_style, newfig=False)
        plt.subplot(122)

        means = 100 * (1 - np.mean(self.trace["CMReduction"], axis=0))
        li = 100 * (1 - np.percentile(self.trace["CMReduction"], 2.5, axis=0))
        ui = 100 * (1 - np.percentile(self.trace["CMReduction"], 97.5, axis=0))
        lq = 100 * (1 - np.percentile(self.trace["CMReduction"], 25, axis=0))
        uq = 100 * (1 - np.percentile(self.trace["CMReduction"], 75, axis=0))

        N_cms = means.size

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

        plt.yticks(
            -np.arange(len(self.d.CMs)),
            [f"     " for f in self.d.CMs]
        )

        ax = plt.gca()
        x_min, x_max = plt.xlim()
        x_r = x_max - x_min
        print(x_r)
        for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
            ticklabel.set_color(self.cm_plot_style[i][1])
            plt.text(x_min - 0.13 * x_r, tickloc, self.cm_plot_style[i][0], horizontalalignment='center',
                     verticalalignment='center',
                     fontproperties=fp2, fontsize=10, color=self.cm_plot_style[i][1])

        plt.xticks(xtick_vals, xtick_str, fontsize=6)
        plt.xlabel("Average Additional Reduction in $R$", fontsize=8)
        plt.tight_layout()

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
            self, data, cm_plot_style=None, name="", model=None
    ):
        super().__init__(data, cm_plot_style, name=name, model=model)

        self.DelayProb = np.array([0.00000000e+00, 1.64635735e-06, 3.15032703e-05, 1.86360977e-04,
                                   6.26527963e-04, 1.54172466e-03, 3.10103643e-03, 5.35663499e-03,
                                   8.33979000e-03, 1.19404848e-02, 1.59939055e-02, 2.03185081e-02,
                                   2.47732062e-02, 2.90464491e-02, 3.30612027e-02, 3.66089026e-02,
                                   3.95642697e-02, 4.18957120e-02, 4.35715814e-02, 4.45816884e-02,
                                   4.49543992e-02, 4.47474142e-02, 4.40036056e-02, 4.27545988e-02,
                                   4.11952870e-02, 3.92608505e-02, 3.71824356e-02, 3.48457206e-02,
                                   3.24845883e-02, 3.00814850e-02, 2.76519177e-02, 2.52792720e-02,
                                   2.30103580e-02, 2.07636698e-02, 1.87005838e-02, 1.67560244e-02,
                                   1.49600154e-02, 1.32737561e-02, 1.17831130e-02, 1.03716286e-02,
                                   9.13757250e-03, 7.98287530e-03, 6.96265658e-03, 6.05951833e-03,
                                   5.26450572e-03, 4.56833017e-03, 3.93189069e-03, 3.38098392e-03,
                                   2.91542076e-03, 2.49468747e-03, 2.13152106e-03, 1.82750115e-03,
                                   1.55693122e-03, 1.31909933e-03, 1.11729819e-03, 9.46588730e-04,
                                   8.06525991e-04, 6.81336089e-04, 5.74623210e-04, 4.80157895e-04,
                                   4.02211774e-04, 3.35345193e-04, 2.82450401e-04, 2.38109993e-04])

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.2

        observed = []
        for r in range(self.nRs):
            skipped_days = []
            for d in range(self.nDs):
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed.append(r * self.nDs + d)
                else:
                    skipped_days.append(d)

            if len(skipped_days) > 0:
                # print(f"Skipped day {[(data.Ds[sk].day, data.Ds[sk].month) for sk in skipped_days]} for {data.Rs[r]}")
                pass

        self.observed_days = np.array(observed)

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)
        self.predict_all_days = True

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2,
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=0.2, mu=np.log(R_hyperprior_mean),
            )

            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.2
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

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction
            )

            serial_interval_sigma = np.sqrt(SI_ALPHA / SI_BETA ** 2)
            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (np.exp(self.ExpectedLogR / si_alpha) - T.ones_like(
                                               self.ExpectedLogR)),
                                           plot_trace=False
                                           )

            self.Growth = pm.Normal("Growth",
                                    self.ExpectedGrowth,
                                    self.DailyGrowthNoise,
                                    shape=(self.nORs, self.nDs))

            self.Growth = pm.Deterministic("Growth",
                                           self.ExpectedGrowth)

            # self.Det("Z1", self.Growth - self.ExpectedGrowth, plot_trace=False)

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

            self.Phi = pm.HalfNormal("Phi", 5)

            self.NewDeaths = pm.Data("NewDeaths",
                                     self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[self.observed_days])

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.observed_days],
                alpha=self.Phi,
                shape=(len(self.observed_days),),
                observed=self.NewDeaths
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

            ec = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=np.repeat(np.array([self.trace.Phi]), nDs, axis=0).T)
            ec_output = dist.random()

            means_expected_deaths, lu_ed, up_ed, err_expected_deaths = produce_CIs(
                ec_output
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
            # z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1[:, country_indx, :])
            # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])

            means_id, lu_id, up_id, err_id = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            plt.plot(days_x, means_id, color="tab:blue", label="R")
            plt.fill_between(
                days_x, lu_id, up_id, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.xlim([min_x, max_x])
            plt.ylim([0, 5])
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$R$")

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
                ax1.legend(*ax.get_legend_handles_labels(), prop={"size": 8}, loc=(0.9, 0.9))
                ax2.legend(prop={"size": 8}, loc="lower left")
                # ax4.legend(lines + lines2, labels + labels2, prop={"size": 8})

class CMActive_Final(BaseCMModel):
    def __init__(
            self, data, cm_plot_style=None, name="", model=None
    ):
        super().__init__(data, cm_plot_style, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProb = np.array([0., 0.0252817, 0.03717965, 0.05181224, 0.06274125,
                                        0.06961334, 0.07277174, 0.07292397, 0.07077184, 0.06694868,
                                        0.06209945, 0.05659917, 0.0508999, 0.0452042, 0.03976573,
                                        0.03470891, 0.0299895, 0.02577721, 0.02199923, 0.01871723,
                                        0.01577148, 0.01326564, 0.01110783, 0.00928827, 0.0077231,
                                        0.00641162, 0.00530572, 0.00437895, 0.00358801, 0.00295791,
                                        0.0024217, 0.00197484])

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.2

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
                        self.d.Confirmed.data[r, d]) and d < (self.nDs - 7):
                    observed.append(r * self.nDs + d)
                else:
                    skipped_days.append(d)
                    self.d.NewCases.mask[r, d] = True

        self.observed_days = np.array(observed)

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2,
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=0.2, mu=np.log(R_hyperprior_mean),
            )

            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.2
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

            serial_interval_sigma = np.sqrt(SI_ALPHA / SI_BETA ** 2)
            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2
            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones_like(
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

class CMCombined_Final_Old(BaseCMModel):
    def __init__(
            self, data, cm_plot_style=None, name="", model=None
    ):
        super().__init__(data, cm_plot_style, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProbCases = np.array([0., 0.0252817, 0.03717965, 0.05181224, 0.06274125,
                                        0.06961334, 0.07277174, 0.07292397, 0.07077184, 0.06694868,
                                        0.06209945, 0.05659917, 0.0508999, 0.0452042, 0.03976573,
                                        0.03470891, 0.0299895, 0.02577721, 0.02199923, 0.01871723,
                                        0.01577148, 0.01326564, 0.01110783, 0.00928827, 0.0077231,
                                        0.00641162, 0.00530572, 0.00437895, 0.00358801, 0.00295791,
                                        0.0024217, 0.00197484])

        self.DelayProbCases = self.DelayProbCases.reshape((1, self.DelayProbCases.size))

        self.DelayProbDeaths = np.array([0.00000000e+00, 1.64635735e-06, 3.15032703e-05, 1.86360977e-04,
                                         6.26527963e-04, 1.54172466e-03, 3.10103643e-03, 5.35663499e-03,
                                         8.33979000e-03, 1.19404848e-02, 1.59939055e-02, 2.03185081e-02,
                                         2.47732062e-02, 2.90464491e-02, 3.30612027e-02, 3.66089026e-02,
                                         3.95642697e-02, 4.18957120e-02, 4.35715814e-02, 4.45816884e-02,
                                         4.49543992e-02, 4.47474142e-02, 4.40036056e-02, 4.27545988e-02,
                                         4.11952870e-02, 3.92608505e-02, 3.71824356e-02, 3.48457206e-02,
                                         3.24845883e-02, 3.00814850e-02, 2.76519177e-02, 2.52792720e-02,
                                         2.30103580e-02, 2.07636698e-02, 1.87005838e-02, 1.67560244e-02,
                                         1.49600154e-02, 1.32737561e-02, 1.17831130e-02, 1.03716286e-02,
                                         9.13757250e-03, 7.98287530e-03, 6.96265658e-03, 6.05951833e-03,
                                         5.26450572e-03, 4.56833017e-03, 3.93189069e-03, 3.38098392e-03,
                                         2.91542076e-03, 2.49468747e-03, 2.13152106e-03, 1.82750115e-03,
                                         1.55693122e-03, 1.31909933e-03, 1.11729819e-03, 9.46588730e-04,
                                         8.06525991e-04, 6.81336089e-04, 5.74623210e-04, 4.80157895e-04,
                                         4.02211774e-04, 3.35345193e-04, 2.82450401e-04, 2.38109993e-04])
        self.DelayProbDeaths = self.DelayProbDeaths.reshape((1, self.DelayProbDeaths.size))

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.1

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
                        self.d.Confirmed.data[r, d]) and d < (self.nDs - 7):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

        self.all_observed_active = np.array(observed_active)

        observed_deaths = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 10 deaths
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        self.all_observed_deaths = np.array(observed_deaths)

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2,
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:
            self.HyperCMVar = pm.HalfStudentT(
                "HyperCMVar", nu=10, sigma=0.1
            )

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, self.HyperCMVar, shape=(self.nCMs,))

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=0.2, mu=np.log(R_hyperprior_mean),
            )

            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.2
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

            serial_interval_sigma = np.sqrt(SI_ALPHA / SI_BETA ** 2)
            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones_like(
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

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nORs,))
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

            # learn the output noise for this.
            self.Phi = pm.HalfNormal("Phi_1", 5)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                alpha=self.Phi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
            )

            self.Z2C = pm.Deterministic(
                "Z2C",
                self.ObservedCases - self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nORs,))
            self.InfectedDeaths_log = pm.Deterministic("InfectedDeaths_log", T.reshape(self.InitialSizeDeaths_log, (
                self.nORs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(self.InfectedDeaths_log))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                alpha=self.Phi,
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
            # dist = pm.NegativeBinomial.dist(mu=ec, alpha=30)
            ec_output = dist.random()

            means_ec, lu_ec, up_ec, err_ec = produce_CIs(
                ec_output
            )

            means_id, lu_id, up_id, err_id = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            dist = pm.NegativeBinomial.dist(mu=ed, alpha=30)
            try:
                ed_output = dist.random()
            except:
                print(region)
                ed_output = ed

            means_ed, lu_ed, up_ed, err_ed = produce_CIs(
                ed_output
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

            means_id, lu_id, up_id, err_id = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )
            # z1C_mean, lu_z1C, up_z1C, err_1C = produce_CIs(self.trace.Z1C[:, country_indx, :])
            # z1D_mean, lu_z1D, up_z1D, err_1D = produce_CIs(self.trace.Z1D[:, country_indx, :])
            # # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            #
            # plt.plot(days_x, z1C_mean, color="tab:purple", label="Growth Noise - Cases")
            # plt.fill_between(
            #     days_x, lu_z1C, up_z1C, alpha=0.25, color="tab:purple", linewidth=0
            # )
            # plt.plot(days_x, z1D_mean, color="tab:purple", label="Growth Noise - Deaths")
            # plt.fill_between(
            #     days_x, lu_z1D, up_z1D, alpha=0.25, color="tab:orange", linewidth=0
            # )
            #
            # plt.xlim([min_x, max_x])
            # plt.ylim([-2, 2])
            # plt.xticks(locs, xlabels, rotation=-30)
            # plt.ylabel("$Z$")

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

    def plot_subset_region_predictions(self, region_indxs, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for i, country_indx in enumerate(region_indxs):

            region = self.d.Rs[country_indx]

            if i % 3 == 0:
                plt.figure(figsize=(10, 11), dpi=300)

            plt.subplot(3, 3, 3 * (i % 3) + 1)

            means_ic, lu_ic, up_ic, err_ic = produce_CIs(
                self.trace.InfectedCases[:, country_indx, :]
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)
            ec_output = dist.random()

            means_ec, lu_ec, up_ec, err_ec = produce_CIs(
                ec_output
            )

            means_id, lu_id, up_id, err_id = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            try:
                ed_output = dist.random()
            except:
                print(region)
                ed_output = ed

            means_ed, lu_ed, up_ed, err_ed = produce_CIs(
                ed_output
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
                label="Estimated New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ec, up_ec, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="New Cases (Smoothed)",
                marker="o",
                s=10,
                color="tab:blue",
                alpha=0.9,
                zorder=3,
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
                label="Estimated New Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="New Deaths (Smoothed)",
                marker="o",
                s=10,
                color="tab:red",
                alpha=0.9,
                zorder=3,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            tick_vals = np.arange(7)
            plt.ylim([10 ** 0, 10 ** 6])
            plt.yticks(np.power(10.0, tick_vals),
                       [f"${np.power(10.0, loc):.0f}$" if loc < 2 else f"$10^{loc}$" for loc in tick_vals])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(3, 3, 3 * (i % 3) + 2)

            ax2 = plt.gca()

            means_g, lu_g, up_g, err_g = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            means_base, lu_base, up_base, err_base = produce_CIs(
                np.exp(self.trace.RegionLogR[:, country_indx])
            )

            plt.plot(days_x, means_g, zorder=1, color="tab:gray", label="$R_{t}$")
            plt.plot([min_x, max_x], [means_base, means_base], "--", zorder=-1, label="$R_0$", color="tab:red",
                     linewidth=0.75)
            # plt.plot(days_x, med_agd, "--", color="tab:orange")

            plt.fill_between(days_x, lu_g, up_g, alpha=0.25, color="tab:gray", linewidth=0)
            plt.fill_between(days_x, lu_base, up_base, alpha=0.15, color="tab:red", linewidth=0, zorder=-1)

            plt.ylim([0, 6])
            plt.xlim([min_x, max_x])
            plt.ylabel("R")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"{self.d.RNames[region][0]}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(3, 3, 3 * (i % 3) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1c_m, lu_z1c, up_z1c, err_z1c = produce_CIs(self.trace.Z1C[:, country_indx, :])
            z1d_m, lu_z1d, up_z1d, err_z1d = produce_CIs(self.trace.Z1D[:, country_indx, :])

            plt.plot(days_x, z1c_m, color="tab:purple", label="$\epsilon^{(C)}$")
            plt.fill_between(days_x, lu_z1c, up_z1c, alpha=0.25, color="tab:purple", linewidth=0)
            plt.plot(days_x, z1d_m, color="tab:orange", label="$\epsilon^{(D)}$")
            plt.fill_between(days_x, lu_z1d, up_z1d, alpha=0.25, color="tab:orange", linewidth=0)
            plt.xlim([min_x, max_x])
            plt.ylim([-0.75, 0.75])
            plt.plot([min_x, max_x], [0, 0], "--", linewidth=0.5, color="k")
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$\epsilon$")

            # ax4.twinx()
            # ax5 = plt.gca()
            #
            # z2c_m, lu_z2c, up_z2c, err_z2c = produce_CIs(self.trace.ExpectedCases[:, country_indx, self.ObservedDaysIndx] - self.d.NewCases.data[country_indx, self.ObservedDaysIndx])
            #
            # plt.plot(self.ObservedDaysIndx, z2c_m, color="tab:orange", label="Cases Output Noise")
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

            if i % 3 == 2 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines3, labels3 = ax4.get_legend_handles_labels()
                ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, prop={"size": 10}, loc=(0.55, 0.6),
                           shadow=True,
                           fancybox=True, ncol=5, bbox_to_anchor=(-1, -0.3))

                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}"
                    )

class CMCombined_Final(BaseCMModel):
    def __init__(
            self, data, cm_plot_style=None, name="", model=None
    ):
        super().__init__(data, cm_plot_style, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProbCases = np.array([0., 0.0252817, 0.03717965, 0.05181224, 0.06274125,
                                        0.06961334, 0.07277174, 0.07292397, 0.07077184, 0.06694868,
                                        0.06209945, 0.05659917, 0.0508999, 0.0452042, 0.03976573,
                                        0.03470891, 0.0299895, 0.02577721, 0.02199923, 0.01871723,
                                        0.01577148, 0.01326564, 0.01110783, 0.00928827, 0.0077231,
                                        0.00641162, 0.00530572, 0.00437895, 0.00358801, 0.00295791,
                                        0.0024217, 0.00197484])

        self.DelayProbCases = self.DelayProbCases.reshape((1, self.DelayProbCases.size))

        self.DelayProbDeaths = np.array([0.00000000e+00, 1.64635735e-06, 3.15032703e-05, 1.86360977e-04,
                                         6.26527963e-04, 1.54172466e-03, 3.10103643e-03, 5.35663499e-03,
                                         8.33979000e-03, 1.19404848e-02, 1.59939055e-02, 2.03185081e-02,
                                         2.47732062e-02, 2.90464491e-02, 3.30612027e-02, 3.66089026e-02,
                                         3.95642697e-02, 4.18957120e-02, 4.35715814e-02, 4.45816884e-02,
                                         4.49543992e-02, 4.47474142e-02, 4.40036056e-02, 4.27545988e-02,
                                         4.11952870e-02, 3.92608505e-02, 3.71824356e-02, 3.48457206e-02,
                                         3.24845883e-02, 3.00814850e-02, 2.76519177e-02, 2.52792720e-02,
                                         2.30103580e-02, 2.07636698e-02, 1.87005838e-02, 1.67560244e-02,
                                         1.49600154e-02, 1.32737561e-02, 1.17831130e-02, 1.03716286e-02,
                                         9.13757250e-03, 7.98287530e-03, 6.96265658e-03, 6.05951833e-03,
                                         5.26450572e-03, 4.56833017e-03, 3.93189069e-03, 3.38098392e-03,
                                         2.91542076e-03, 2.49468747e-03, 2.13152106e-03, 1.82750115e-03,
                                         1.55693122e-03, 1.31909933e-03, 1.11729819e-03, 9.46588730e-04,
                                         8.06525991e-04, 6.81336089e-04, 5.74623210e-04, 4.80157895e-04,
                                         4.02211774e-04, 3.35345193e-04, 2.82450401e-04, 2.38109993e-04])
        self.DelayProbDeaths = self.DelayProbDeaths.reshape((1, self.DelayProbDeaths.size))

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.2

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
                        self.d.Confirmed.data[r, d]) and d < (self.nDs - 7):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

        self.all_observed_active = np.array(observed_active)

        observed_deaths = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 10 deaths
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        self.all_observed_deaths = np.array(observed_deaths)

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2,
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:
            self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=0.2, mu=np.log(R_hyperprior_mean),
            )

            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.2
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

            serial_interval_sigma = np.sqrt(SI_ALPHA / SI_BETA ** 2)
            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones_like(
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

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nORs,))
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

            # learn the output noise for this.
            self.Phi = pm.HalfNormal("Phi_1", 5)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                alpha=self.Phi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
            )

            self.Z2C = pm.Deterministic(
                "Z2C",
                self.ObservedCases - self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nORs,))
            self.InfectedDeaths_log = pm.Deterministic("InfectedDeaths_log", T.reshape(self.InitialSizeDeaths_log, (
                self.nORs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(self.InfectedDeaths_log))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                alpha=self.Phi,
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
            # dist = pm.NegativeBinomial.dist(mu=ec, alpha=30)
            ec_output = dist.random()

            means_ec, lu_ec, up_ec, err_ec = produce_CIs(
                ec_output
            )

            means_id, lu_id, up_id, err_id = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            dist = pm.NegativeBinomial.dist(mu=ed, alpha=30)
            try:
                ed_output = dist.random()
            except:
                print(region)
                ed_output = ed

            means_ed, lu_ed, up_ed, err_ed = produce_CIs(
                ed_output
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

            means_id, lu_id, up_id, err_id = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )
            # z1C_mean, lu_z1C, up_z1C, err_1C = produce_CIs(self.trace.Z1C[:, country_indx, :])
            # z1D_mean, lu_z1D, up_z1D, err_1D = produce_CIs(self.trace.Z1D[:, country_indx, :])
            # # z2_mean, lu_z2, up_z2, err_2 = produce_CIs(self.trace.Z2[:, country_indx, :])
            #
            # plt.plot(days_x, z1C_mean, color="tab:purple", label="Growth Noise - Cases")
            # plt.fill_between(
            #     days_x, lu_z1C, up_z1C, alpha=0.25, color="tab:purple", linewidth=0
            # )
            # plt.plot(days_x, z1D_mean, color="tab:purple", label="Growth Noise - Deaths")
            # plt.fill_between(
            #     days_x, lu_z1D, up_z1D, alpha=0.25, color="tab:orange", linewidth=0
            # )
            #
            # plt.xlim([min_x, max_x])
            # plt.ylim([-2, 2])
            # plt.xticks(locs, xlabels, rotation=-30)
            # plt.ylabel("$Z$")

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

    def plot_subset_region_predictions(self, region_indxs, plot_style, save_fig=True, output_dir="./out"):
        assert self.trace is not None

        for i, country_indx in enumerate(region_indxs):

            region = self.d.Rs[country_indx]

            if i % 3 == 0:
                plt.figure(figsize=(10, 11), dpi=300)

            plt.subplot(3, 3, 3 * (i % 3) + 1)

            means_ic, lu_ic, up_ic, err_ic = produce_CIs(
                self.trace.InfectedCases[:, country_indx, :]
            )

            ec = self.trace.ExpectedCases[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)
            ec_output = dist.random()

            means_ec, lu_ec, up_ec, err_ec = produce_CIs(
                ec_output
            )

            means_id, lu_id, up_id, err_id = produce_CIs(
                self.trace.InfectedDeaths[:, country_indx, :]
            )

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            try:
                ed_output = dist.random()
            except:
                print(region)
                ed_output = ed

            means_ed, lu_ed, up_ed, err_ed = produce_CIs(
                ed_output
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
                label="Estimated New Cases",
                zorder=2,
                color="tab:blue"
            )

            plt.fill_between(
                days_x, lu_ec, up_ec, alpha=0.25, color="tab:blue", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                newcases[self.ObservedDaysIndx],
                label="New Cases (Smoothed)",
                marker="o",
                s=10,
                color="tab:blue",
                alpha=0.9,
                zorder=3,
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
                label="Estimated New Deaths",
                zorder=2,
                color="tab:red"
            )

            plt.fill_between(
                days_x, lu_ed, up_ed, alpha=0.25, color="tab:red", linewidth=0
            )

            plt.scatter(
                self.ObservedDaysIndx,
                deaths[self.ObservedDaysIndx],
                label="New Deaths (Smoothed)",
                marker="o",
                s=10,
                color="tab:red",
                alpha=0.9,
                zorder=3,
            )

            ax.set_yscale("log")
            plt.xlim([min_x, max_x])
            tick_vals = np.arange(7)
            plt.ylim([10 ** 0, 10 ** 6])
            plt.yticks(np.power(10.0, tick_vals),
                       [f"${np.power(10.0, loc):.0f}$" if loc < 2 else f"$10^{loc}$" for loc in tick_vals])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(3, 3, 3 * (i % 3) + 2)

            ax2 = plt.gca()

            means_g, lu_g, up_g, err_g = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            means_base, lu_base, up_base, err_base = produce_CIs(
                np.exp(self.trace.RegionLogR[:, country_indx])
            )

            plt.plot(days_x, means_g, zorder=1, color="tab:gray", label="$R_{t}$")
            plt.plot([min_x, max_x], [means_base, means_base], "--", zorder=-1, label="$R_0$", color="tab:red",
                     linewidth=0.75)
            # plt.plot(days_x, med_agd, "--", color="tab:orange")

            plt.fill_between(days_x, lu_g, up_g, alpha=0.25, color="tab:gray", linewidth=0)
            plt.fill_between(days_x, lu_base, up_base, alpha=0.15, color="tab:red", linewidth=0, zorder=-1)

            plt.ylim([0, 6])
            plt.xlim([min_x, max_x])
            plt.ylabel("R")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"{self.d.RNames[region][0]}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(3, 3, 3 * (i % 3) + 3)
            axis_scale = 1.5
            ax4 = plt.gca()
            z1c_m, lu_z1c, up_z1c, err_z1c = produce_CIs(self.trace.Z1C[:, country_indx, :])
            z1d_m, lu_z1d, up_z1d, err_z1d = produce_CIs(self.trace.Z1D[:, country_indx, :])

            plt.plot(days_x, z1c_m, color="tab:purple", label="$\epsilon^{(C)}$")
            plt.fill_between(days_x, lu_z1c, up_z1c, alpha=0.25, color="tab:purple", linewidth=0)
            plt.plot(days_x, z1d_m, color="tab:orange", label="$\epsilon^{(D)}$")
            plt.fill_between(days_x, lu_z1d, up_z1d, alpha=0.25, color="tab:orange", linewidth=0)
            plt.xlim([min_x, max_x])
            plt.ylim([-0.75, 0.75])
            plt.plot([min_x, max_x], [0, 0], "--", linewidth=0.5, color="k")
            plt.xticks(locs, xlabels, rotation=-30)
            plt.ylabel("$\epsilon$")

            # ax4.twinx()
            # ax5 = plt.gca()
            #
            # z2c_m, lu_z2c, up_z2c, err_z2c = produce_CIs(self.trace.ExpectedCases[:, country_indx, self.ObservedDaysIndx] - self.d.NewCases.data[country_indx, self.ObservedDaysIndx])
            #
            # plt.plot(self.ObservedDaysIndx, z2c_m, color="tab:orange", label="Cases Output Noise")
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

            if i % 3 == 2 or country_indx == len(self.d.Rs) - 1:
                plt.tight_layout()
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines3, labels3 = ax4.get_legend_handles_labels()
                ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, prop={"size": 10}, loc=(0.55, 0.6),
                           shadow=True,
                           fancybox=True, ncol=5, bbox_to_anchor=(-1, -0.3))

                if save_fig:
                    save_fig_pdf(
                        output_dir,
                        f"CountryPredictionPlot{((country_indx + 1) / 5):.1f}"
                    )


# ICL Model versions - not used for our results
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

        self.SI_rev = self.SI[::-1].reshape((1, self.SI.size))
        # infection --> confirmed delay
        self.DelayProb = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                   0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                   0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                   0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                   0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                   0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.2

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

        observed = []
        not_observed = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Confirmed.data[r, d]) and d < self.nDs - 7:
                    observed.append(r * self.nDs + d)
                else:
                    # set the mask to False!
                    self.d.NewCases.mask[r, d] = False
                    not_observed.append(r * self.nDs + d)

        self.observed_days = np.array(observed)
        self.not_observed = np.array(not_observed)

    def build_model(self):
        with self.model:
            self.HyperCMVar = pm.HalfStudentT(
                "HyperCMVar", nu=10, sigma=0.3
            )

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, self.HyperCMVar, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=1, mu=np.log(3.5),
            )

            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.3
            )

            self.RegionLogR = pm.Normal("RegionLogR", np.log(3.5),
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

            self.InitialSize_log = pm.Normal("InitialSize_log", 0, 20, shape=(self.nORs,))

            # # conv padding
            filter_size = self.SI_rev.size
            conv_padding = 7

            infected = T.zeros((self.nORs, self.nDs + self.SI_rev.size))
            infected = T.set_subtensor(infected[:, (filter_size - conv_padding):filter_size],
                                       pm.math.exp(
                                           self.InitialSize_log.reshape((self.nORs, 1)).repeat(conv_padding, axis=1)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, d].reshape((self.nORs, 1)) * infected[:, d:d + filter_size] * self.SI_rev, axis=1)
                infected = T.set_subtensor(infected[:, d + filter_size], val)

            self.Infected = pm.Deterministic("Infected", infected[:, filter_size:])

            expected_confirmed = C.conv2d(
                self.Infected,
                np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_confirmed.reshape(
                (self.nORs, self.nDs)))

            self.Phi = 25

            self.NewCases = pm.Data("NewCases",
                                    self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.observed_days])
            # self.NewCases = pm.Data("NewCases", self.d.NewCases)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.observed_days],
                alpha=self.Phi,
                shape=(len(self.observed_days),),
                observed=self.NewCases
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
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=6)
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

            newcases_plot = copy.deepcopy(newcases)
            newcases_plot[newcases_plot < 1] = 1e-1

            plt.scatter(
                self.ObservedDaysIndx,
                newcases_plot[self.ObservedDaysIndx].data,
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
            ax.set_axisbelow(True)
            ax.set_zorder(-3)
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -2, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)
            ax1.set_axisbelow(True)
            ax1.set_zorder(-3)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            mean_R = np.mean(np.exp(self.trace.RegionLogR[:, country_indx]))

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.LogR[:, country_indx, :])
            )

            plt.plot([min_x, max_x], [mean_R, mean_R], color="tab:red", linewidth=0.5)
            plt.plot(days_x, means_growth, label="Expected R", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted R", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            mean_R = np.mean(np.exp(self.trace.RegionLogR[:, country_indx]))
            plt.plot([min_x, max_x], [mean_R, mean_R], color="tab:red", linewidth=0.5)
            plt.ylim([0, 10])
            plt.xlim([min_x, max_x])
            plt.ylabel("R")
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


class CMDeath_Final_ICL(BaseCMModel):
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

        self.SI_rev = self.SI[::-1].reshape((1, self.SI.size))
        # infection --> confirmed delay
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
        self.DailyGrowthNoise = 0.2

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

        observed = []
        not_observed = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed.append(r * self.nDs + d)
                else:
                    # set the mask to False!
                    self.d.NewDeaths.mask[r, d] = False
                    not_observed.append(r * self.nDs + d)

        self.observed_days = np.array(observed)
        self.not_observed = np.array(not_observed)

    def build_model(self):
        with self.model:
            self.HyperCMVar = pm.HalfStudentT(
                "HyperCMVar", nu=10, sigma=0.1
            )

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, self.HyperCMVar, shape=(self.nCMs,))
            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRMean = pm.StudentT(
                "HyperRMean", nu=10, sigma=0.2, mu=np.log(3.25),
            )

            self.HyperRVar = pm.HalfStudentT(
                "HyperRVar", nu=10, sigma=0.2
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

            # self.Normal(
            #     "LogR",
            #     self.ExpectedLogR,
            #     self.DailyGrowthNoise,
            #     shape=(self.nORs, self.nDs),
            #     plot_trace=False,
            # )

            self.Det("LogR", self.ExpectedLogR)

            self.Det("Z1", self.LogR - self.ExpectedLogR, plot_trace=False)

            self.InitialSize_log = pm.Normal("InitialSize_log", -4, 30, shape=(self.nORs,))

            # # conv padding
            filter_size = self.SI_rev.size
            conv_padding = 7

            infected = T.zeros((self.nORs, self.nDs + self.SI_rev.size))
            infected = T.set_subtensor(infected[:, (filter_size - conv_padding):filter_size],
                                       pm.math.exp(
                                           self.InitialSize_log.reshape((self.nORs, 1)).repeat(conv_padding, axis=1)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, d].reshape((self.nORs, 1)) * infected[:, d:d + filter_size] * self.SI_rev, axis=1)
                infected = T.set_subtensor(infected[:, d + filter_size], val)

            self.Infected = pm.Deterministic("Infected", infected[:, filter_size:])

            expected_deaths = C.conv2d(
                self.Infected,
                np.reshape(self.DelayProb, newshape=(1, self.DelayProb.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            self.Phi = pm.HalfNormal("Phi", 5)

            self.NewDeaths = pm.Data("NewDeaths",
                                     self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[self.observed_days])

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.observed_days],
                alpha=15,
                shape=(len(self.observed_days),),
                observed=self.NewDeaths
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

            ec = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ec.shape
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=40)

            try:
                ec_output = dist.random()
            except:
                ec_output = ec

            means_ea, lu_ea, up_ea, err_eea = produce_CIs(
                ec_output
            )

            days = self.d.Ds
            days_x = np.arange(len(days))

            min_x = 25
            max_x = len(days) - 1

            newcases = self.d.NewDeaths[country_indx, :]

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

            newcases_plot = copy.deepcopy(newcases)
            newcases_plot[newcases_plot < 1] = 1e-1

            plt.scatter(
                self.ObservedDaysIndx,
                newcases_plot[self.ObservedDaysIndx].data,
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
            ax.set_axisbelow(True)
            ax.set_zorder(-3)
            plt.xlim([min_x, max_x])
            plt.ylim([10 ** -2, 10 ** 5])
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            ax1 = add_cms_to_plot(ax, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)
            ax1.set_axisbelow(True)
            ax1.set_zorder(-3)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 2)

            ax2 = plt.gca()

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.LogR[:, country_indx, :])
            )

            mean_R = np.mean(np.exp(self.trace.RegionLogR[:, country_indx]))
            plt.plot([min_x, max_x], [mean_R, mean_R], color="tab:red", linewidth=0.5)
            plt.plot(days_x, means_growth, label="Expected R", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="Predicted R", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0, 7])
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
                np.repeat(self.d.NewDeaths[country_indx, :].data.reshape(1, nDs), nS,
                          axis=0) - self.trace.ExpectedDeaths[
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


class CMCombined_Final_ICL(BaseCMModel):
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

        self.SI_rev = self.SI[::-1].reshape((1, self.SI.size))
        # infection --> confirmed delay
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

        self.DelayProbCases = np.array([0.00509233, 0.02039664, 0.03766875, 0.0524391, 0.06340527,
                                        0.07034326, 0.07361858, 0.07378182, 0.07167229, 0.06755999,
                                        0.06275661, 0.05731038, 0.05141595, 0.04565263, 0.04028695,
                                        0.03502109, 0.03030662, 0.02611754, 0.02226727, 0.0188904,
                                        0.01592167, 0.01342368, 0.01127307, 0.00934768, 0.00779801,
                                        0.00645582, 0.00534967, 0.00442695])

        self.CMDelayCut = 30
        self.DailyGrowthNoise = 0.2

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
                        self.d.Confirmed.data[r, d]) and d < (self.nDs - 7):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

        self.all_observed_active = np.array(observed_active)

        observed_deaths = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 10 deaths
                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        self.all_observed_deaths = np.array(observed_deaths)

    def build_model(self):
        with self.model:
            self.HyperCMVar = pm.HalfStudentT(
                "HyperCMVar", nu=10, sigma=0.3
            )

            self.CM_Alpha = pm.Normal("CM_Alpha", 0, self.HyperCMVar, shape=(self.nCMs,))
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
                "LogRCases",
                self.ExpectedLogR,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs),
                plot_trace=False,
            )

            # self.Normal(
            #     "LogRDeaths",
            #     self.ExpectedLogR,
            #     self.DailyGrowthNoise,
            #     shape=(self.nORs, self.nDs),
            #     plot_trace=False,
            # )

            self.LogRDeaths = pm.Deterministic("LogRDeaths", self.ExpectedLogR, )

            self.Det("Z1C", self.LogRCases - self.ExpectedLogR, plot_trace=False)
            self.Det("Z1D", self.LogRDeaths - self.ExpectedLogR, plot_trace=False)

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 1, 10, shape=(self.nORs,))
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", -4, 10, shape=(self.nORs,))

            # # conv padding
            filter_size = self.SI_rev.size
            conv_padding = 7

            infected_cases = T.zeros((self.nORs, self.nDs + self.SI_rev.size))
            infected_cases = T.set_subtensor(infected_cases[:, (filter_size - conv_padding):filter_size],
                                             pm.math.exp(self.InitialSizeCases_log.reshape((self.nORs, 1)).repeat(
                                                 conv_padding, axis=1)))
            infected_deaths = T.zeros((self.nORs, self.nDs + self.SI_rev.size))
            infected_deaths = T.set_subtensor(infected_deaths[:, (filter_size - conv_padding):filter_size],
                                              pm.math.exp(self.InitialSizeDeaths_log.reshape((self.nORs, 1)).repeat(
                                                  conv_padding, axis=1)))

            # R is a lognorm
            R_cases = pm.math.exp(self.LogRCases)
            R_deaths = pm.math.exp(self.LogRDeaths)

            for d in range(self.nDs):
                val_c = pm.math.sum(
                    R_cases[:, d].reshape((self.nORs, 1)) * infected_cases[:, d:d + filter_size] * self.SI_rev,
                    axis=1)
                val_d = pm.math.sum(
                    R_deaths[:, d].reshape((self.nORs, 1)) * infected_deaths[:, d:d + filter_size] * self.SI_rev,
                    axis=1)
                infected_deaths = T.set_subtensor(infected_deaths[:, d + filter_size], val_d)
                infected_cases = T.set_subtensor(infected_cases[:, d + filter_size], val_c)

            self.InfectedCases = pm.Deterministic("InfectedCases", infected_cases[:, filter_size:])
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", infected_deaths[:, filter_size:])

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            expected_cases = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbCases, newshape=(1, self.DelayProbCases.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nORs, self.nDs)))

            self.Phi = 25

            self.NewCases = pm.Data("NewCases",
                                    self.d.NewCases.data.reshape((self.nORs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data("NewDeaths",
                                     self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[
                                         self.all_observed_deaths])

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                alpha=self.Phi,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )

            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                alpha=self.Phi,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
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
            # dist = pm.NegativeBinomial.dist(mu=ec + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)
            dist = pm.NegativeBinomial.dist(mu=ec, alpha=25)
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

            dist = pm.NegativeBinomial.dist(mu=ed, alpha=25)
            try:
                ed_output = dist.random()
            except:
                print(region)
                ed_output = ed

            means_ed, lu_ed, up_ed, err_ed = produce_CIs(
                ed_output
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

            mean_R = np.mean(np.exp(self.trace.RegionLogR[:, country_indx]))

            means_growth, lu_g, up_g, err = produce_CIs(
                np.exp(self.trace.ExpectedLogR[:, country_indx, :])
            )

            actual_growth, lu_ag, up_ag, err_act = produce_CIs(
                np.exp(self.trace.LogRCases[:, country_indx, :])
            )

            plt.plot([min_x, max_x], [mean_R, mean_R], color="tab:red", linewidth=0.5)
            plt.plot(days_x, means_growth, label="R Deaths", zorder=1, color="tab:orange")
            plt.plot(days_x, actual_growth, label="R Cases", zorder=1, color="tab:blue")

            plt.fill_between(
                days_x, lu_g, up_g, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.fill_between(
                days_x, lu_ag, up_ag, alpha=0.25, color="tab:blue", linewidth=0
            )
            plt.plot([min_x, max_x], [1, 1], "--", linewidth=0.5, color="lightgrey")

            plt.ylim([0, 7])
            plt.xlim([min_x, max_x])
            plt.ylabel("R")
            locs = np.arange(min_x, max_x, 7)
            xlabels = [f"{days[ts].day}-{days[ts].month}" for ts in locs]
            plt.xticks(locs, xlabels, rotation=-30)
            plt.title(f"Region {region}")
            ax3 = add_cms_to_plot(ax2, self.d.ActiveCMs, country_indx, min_x, max_x, days, plot_style)

            plt.subplot(5, 3, 3 * (country_indx % 5) + 3)
            ax4 = plt.gca()
            z1_mean, lu_z1, up_z1, err_1 = produce_CIs(self.trace.Z1C[:, country_indx, :])
            nS, nDs = self.trace.Z1C[:, country_indx, :].shape
            z2_mean, lu_z2, up_z2, err_2 = produce_CIs(
                np.repeat(self.d.NewCases[country_indx, :].data.reshape(1, nDs), nS,
                          axis=0) - self.trace.ExpectedCases[
                                    :, country_indx, :])
            z3_mean, lu_z3, up_z3, err_3 = produce_CIs(
                np.repeat(self.d.NewDeaths[country_indx, :].data.reshape(1, nDs), nS,
                          axis=0) - self.trace.ExpectedDeaths[
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
            plt.plot(np.arange(self.nDs), z2_mean, color="tab:orange", label="Cases Noise")
            plt.fill_between(
                np.arange(self.nDs), lu_z2, up_z2, alpha=0.25, color="tab:orange", linewidth=0
            )

            plt.plot(np.arange(self.nDs), z3_mean, color="tab:orange", label="Death Noise")
            plt.fill_between(
                np.arange(self.nDs), lu_z3, up_z3, alpha=0.25, color="tab:orange", linewidth=0
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
