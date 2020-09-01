import copy
import logging
import os
from datetime import datetime

import numpy as np
import pymc3 as pm
import seaborn as sns
import theano.tensor as T
import theano.tensor.signal.conv as C
from pymc3 import Model

log = logging.getLogger(__name__)
sns.set_style("ticks")

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")

# cereda mean, eurosurveilance SI
SI_ALPHA = 7.935
SI_BETA = 1.188


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


class BaseCMModelLegacy(Model):
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
        li = 100 * (1 - np.percentile(self.trace["CMReduction"], 5, axis=0))
        ui = 100 * (1 - np.percentile(self.trace["CMReduction"], 95, axis=0))
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
            [f"{f}" for f in self.d.CMs]
        )

        ax = plt.gca()
        x_min, x_max = plt.xlim()
        x_r = x_max - x_min
        # print(x_r)
        # for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
        #     ticklabel.set_color(self.cm_plot_style[i][1])
        #     plt.text(x_min - 0.13 * x_r, tickloc, self.cm_plot_style[i][0], horizontalalignment='center',
        #              verticalalignment='center',
        #              fontproperties=fp2, fontsize=10, color=self.cm_plot_style[i][1])

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
            self.trace = pm.sample(N, chains=chains, cores=cores, init="jitter+adapt_diag", target_accept=0.8,
                                   max_treedepth=12, **kwargs)


class CMCombined_FinalLegacy(BaseCMModelLegacy):
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

        self.DelayProbDeaths = np.array([0.00000000e+00, 2.24600347e-06, 3.90382088e-05, 2.34307085e-04,
                                         7.83555003e-04, 1.91221622e-03, 3.78718437e-03, 6.45923913e-03,
                                         9.94265709e-03, 1.40610714e-02, 1.86527920e-02, 2.34311421e-02,
                                         2.81965055e-02, 3.27668001e-02, 3.68031574e-02, 4.03026198e-02,
                                         4.30521951e-02, 4.50637136e-02, 4.63315047e-02, 4.68794406e-02,
                                         4.67334059e-02, 4.59561441e-02, 4.47164503e-02, 4.29327455e-02,
                                         4.08614522e-02, 3.85082076e-02, 3.60294203e-02, 3.34601703e-02,
                                         3.08064505e-02, 2.81766028e-02, 2.56165924e-02, 2.31354369e-02,
                                         2.07837267e-02, 1.86074383e-02, 1.65505661e-02, 1.46527043e-02,
                                         1.29409383e-02, 1.13695920e-02, 9.93233881e-03, 8.66063386e-03,
                                         7.53805464e-03, 6.51560047e-03, 5.63512264e-03, 4.84296166e-03,
                                         4.14793478e-03, 3.56267297e-03, 3.03480656e-03, 2.59406730e-03,
                                         2.19519042e-03, 1.85454286e-03, 1.58333238e-03, 1.33002321e-03,
                                         1.11716435e-03, 9.35360376e-04, 7.87780158e-04, 6.58601602e-04,
                                         5.48147154e-04, 4.58151351e-04, 3.85878963e-04, 3.21623249e-04,
                                         2.66129174e-04, 2.21364768e-04, 1.80736566e-04, 1.52350196e-04])
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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2, cm_prior='normal',
                    serial_interval_mean=SI_ALPHA / SI_BETA, serial_interval_sigma=np.sqrt(SI_ALPHA / SI_BETA ** 2),
                    conf_noise=None, deaths_noise=None
                    ):
        with self.model:
            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'icl':
                self.CM_Alpha_t = pm.Gamma("CM_Alpha_t", 1 / 6, 1, shape=(self.nCMs,))
                self.CM_Alpha = pm.Deterministic("CM_Alpha", self.CM_Alpha_t - np.log(1.05) / 6)

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic("RegionR", R_hyperprior_mean + self.RegionLogR_noise * self.HyperRVar)

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
                T.reshape(pm.math.log(self.RegionR), (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones((self.nORs, self.nDs))),
                                           plot_trace=False
                                           )

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, self.DailyGrowthNoise, shape=(self.nORs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, self.DailyGrowthNoise,
                                               shape=(self.nORs, self.nDs))

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth + self.GrowthCasesNoise)
            self.GrowthDeaths = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth + self.GrowthDeathsNoise)

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nORs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

            expected_cases = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbCases, newshape=(1, self.DelayProbCases.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed conf noise
            if conf_noise is None:
                # learn the output noise for this
                self.Phi = pm.HalfNormal("Phi_1", 5)

                # effectively handle missing values ourselves
                self.ObservedCases = pm.NegativeBinomial(
                    "ObservedCases",
                    mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                    alpha=self.Phi,
                    shape=(len(self.all_observed_active),),
                    observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
                )

            else:
                # effectively handle missing values ourselves
                self.ObservedCases = pm.NegativeBinomial(
                    "ObservedCases",
                    mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                    alpha=conf_noise,
                    shape=(len(self.all_observed_active),),
                    observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
                )

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nORs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed deaths noise
            if deaths_noise is None:
                if conf_noise is not None:
                    # learn the output noise for this
                    self.Phi = pm.HalfNormal("Phi_1", 5)

                # effectively handle missing values ourselves
                self.ObservedDeaths = pm.NegativeBinomial(
                    "ObservedDeaths",
                    mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                    alpha=self.Phi,
                    shape=(len(self.all_observed_deaths),),
                    observed=self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]
                )
            else:
                # effectively handle missing values ourselves
                self.ObservedDeaths = pm.NegativeBinomial(
                    "ObservedDeaths",
                    mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                    alpha=deaths_noise,
                    shape=(len(self.all_observed_deaths),),
                    observed=self.d.NewDeaths.data[:, self.CMDelayCut:].reshape((self.nORs * self.nDs,))[
                        self.all_observed_deaths]
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

    def plot_subset_region_predictions(self, region_indxs, plot_style, n_rows=3, fig_height=11, save_fig=True,
                                       output_dir="./out"):
        assert self.trace is not None

        for i, country_indx in enumerate(region_indxs):

            region = self.d.Rs[country_indx]

            if i % n_rows == 0:
                plt.figure(figsize=(10, fig_height), dpi=300)

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 1)

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

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            ids = self.trace.InfectedDeaths[:, country_indx, :]
            try:
                ed_output = dist.random()
            except:
                print("hi?")
                print(region)
                ed_output = np.ones_like(ids) * 10 ** -5
                ids = np.ones_like(ids) * 10 ** -5

            # if np.isnan(self.d.Deaths.data[country_indx, -1]):
            #     ed_output = np.ones_like(ids) * 10 ** -5
            #     ids = np.ones_like(ids) * 10 ** -5

            means_id, lu_id, up_id, err_id = produce_CIs(
                ids
            )

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

            plt.scatter(
                self.ObservedDaysIndx,
                newcases.data[self.ObservedDaysIndx],
                label="New Cases (Smoothed)",
                marker="o",
                s=10,
                color="tab:blue",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.scatter(
                self.ObservedDaysIndx,
                deaths.data[self.ObservedDaysIndx],
                label="New Deaths (Smoothed)",
                marker="o",
                s=10,
                color="tab:red",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 2)

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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 3)
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

            if i % n_rows == (n_rows - 1) or country_indx == len(self.d.Rs) - 1:
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
                        f"Fits{((country_indx + 1) / 5):.1f}"
                    )


class CMCombined_FinalLegacyFixedDispersion(BaseCMModelLegacy):
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

        self.DelayProbDeaths = np.array([0.00000000e+00, 2.24600347e-06, 3.90382088e-05, 2.34307085e-04,
                                         7.83555003e-04, 1.91221622e-03, 3.78718437e-03, 6.45923913e-03,
                                         9.94265709e-03, 1.40610714e-02, 1.86527920e-02, 2.34311421e-02,
                                         2.81965055e-02, 3.27668001e-02, 3.68031574e-02, 4.03026198e-02,
                                         4.30521951e-02, 4.50637136e-02, 4.63315047e-02, 4.68794406e-02,
                                         4.67334059e-02, 4.59561441e-02, 4.47164503e-02, 4.29327455e-02,
                                         4.08614522e-02, 3.85082076e-02, 3.60294203e-02, 3.34601703e-02,
                                         3.08064505e-02, 2.81766028e-02, 2.56165924e-02, 2.31354369e-02,
                                         2.07837267e-02, 1.86074383e-02, 1.65505661e-02, 1.46527043e-02,
                                         1.29409383e-02, 1.13695920e-02, 9.93233881e-03, 8.66063386e-03,
                                         7.53805464e-03, 6.51560047e-03, 5.63512264e-03, 4.84296166e-03,
                                         4.14793478e-03, 3.56267297e-03, 3.03480656e-03, 2.59406730e-03,
                                         2.19519042e-03, 1.85454286e-03, 1.58333238e-03, 1.33002321e-03,
                                         1.11716435e-03, 9.35360376e-04, 7.87780158e-04, 6.58601602e-04,
                                         5.48147154e-04, 4.58151351e-04, 3.85878963e-04, 3.21623249e-04,
                                         2.66129174e-04, 2.21364768e-04, 1.80736566e-04, 1.52350196e-04])
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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2, cm_prior='normal',
                    serial_interval_mean=SI_ALPHA / SI_BETA, serial_interval_sigma=np.sqrt(SI_ALPHA / SI_BETA ** 2),
                    conf_noise=None, deaths_noise=None
                    ):
        with self.model:
            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'icl':
                self.CM_Alpha_t = pm.Gamma("CM_Alpha_t", 1 / 6, 1, shape=(self.nCMs,))
                self.CM_Alpha = pm.Deterministic("CM_Alpha", self.CM_Alpha_t - np.log(1.05) / 6)

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic("RegionR", R_hyperprior_mean + self.RegionLogR_noise * self.HyperRVar)

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
                T.reshape(pm.math.log(self.RegionR), (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones((self.nORs, self.nDs))),
                                           plot_trace=False
                                           )

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, self.DailyGrowthNoise, shape=(self.nORs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, self.DailyGrowthNoise,
                                               shape=(self.nORs, self.nDs))

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth + self.GrowthCasesNoise)
            self.GrowthDeaths = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth + self.GrowthDeathsNoise)

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nORs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

            expected_cases = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbCases, newshape=(1, self.DelayProbCases.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed conf noise
            if conf_noise is None:
                # learn the output noise for this
                self.Phi = pm.HalfNormal("Phi_1", 5)

                # effectively handle missing values ourselves
                self.ObservedCases = pm.NegativeBinomial(
                    "ObservedCases",
                    mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                    alpha=60.0,
                    shape=(len(self.all_observed_active),),
                    observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
                )

            else:
                # effectively handle missing values ourselves
                self.ObservedCases = pm.NegativeBinomial(
                    "ObservedCases",
                    mu=self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active],
                    alpha=60.0,
                    shape=(len(self.all_observed_active),),
                    observed=self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active]
                )

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nORs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed deaths noise
            if deaths_noise is None:
                if conf_noise is not None:
                    # learn the output noise for this
                    self.Phi = pm.HalfNormal("Phi_1", 5)

                # effectively handle missing values ourselves
                self.ObservedDeaths = pm.NegativeBinomial(
                    "ObservedDeaths",
                    mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                    alpha=60.0,
                    shape=(len(self.all_observed_deaths),),
                    observed=self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]
                )
            else:
                # effectively handle missing values ourselves
                self.ObservedDeaths = pm.NegativeBinomial(
                    "ObservedDeaths",
                    mu=self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths],
                    alpha=60.0,
                    shape=(len(self.all_observed_deaths),),
                    observed=self.d.NewDeaths.data[:, self.CMDelayCut:].reshape((self.nORs * self.nDs,))[
                        self.all_observed_deaths]
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

    def plot_subset_region_predictions(self, region_indxs, plot_style, n_rows=3, fig_height=11, save_fig=True,
                                       output_dir="./out"):
        assert self.trace is not None

        for i, country_indx in enumerate(region_indxs):

            region = self.d.Rs[country_indx]

            if i % n_rows == 0:
                plt.figure(figsize=(10, fig_height), dpi=300)

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 1)

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

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            ids = self.trace.InfectedDeaths[:, country_indx, :]
            try:
                ed_output = dist.random()
            except:
                print("hi?")
                print(region)
                ed_output = np.ones_like(ids) * 10 ** -5
                ids = np.ones_like(ids) * 10 ** -5

            # if np.isnan(self.d.Deaths.data[country_indx, -1]):
            #     ed_output = np.ones_like(ids) * 10 ** -5
            #     ids = np.ones_like(ids) * 10 ** -5

            means_id, lu_id, up_id, err_id = produce_CIs(
                ids
            )

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

            plt.scatter(
                self.ObservedDaysIndx,
                newcases.data[self.ObservedDaysIndx],
                label="New Cases (Smoothed)",
                marker="o",
                s=10,
                color="tab:blue",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.scatter(
                self.ObservedDaysIndx,
                deaths.data[self.ObservedDaysIndx],
                label="New Deaths (Smoothed)",
                marker="o",
                s=10,
                color="tab:red",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 2)

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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 3)
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

            if i % n_rows == (n_rows - 1) or country_indx == len(self.d.Rs) - 1:
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
                        f"Fits{((country_indx + 1) / 5):.1f}"
                    )


class CMCombined_FinalLegacyLognorm(BaseCMModelLegacy):
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

        self.DelayProbDeaths = np.array([0.00000000e+00, 2.24600347e-06, 3.90382088e-05, 2.34307085e-04,
                                         7.83555003e-04, 1.91221622e-03, 3.78718437e-03, 6.45923913e-03,
                                         9.94265709e-03, 1.40610714e-02, 1.86527920e-02, 2.34311421e-02,
                                         2.81965055e-02, 3.27668001e-02, 3.68031574e-02, 4.03026198e-02,
                                         4.30521951e-02, 4.50637136e-02, 4.63315047e-02, 4.68794406e-02,
                                         4.67334059e-02, 4.59561441e-02, 4.47164503e-02, 4.29327455e-02,
                                         4.08614522e-02, 3.85082076e-02, 3.60294203e-02, 3.34601703e-02,
                                         3.08064505e-02, 2.81766028e-02, 2.56165924e-02, 2.31354369e-02,
                                         2.07837267e-02, 1.86074383e-02, 1.65505661e-02, 1.46527043e-02,
                                         1.29409383e-02, 1.13695920e-02, 9.93233881e-03, 8.66063386e-03,
                                         7.53805464e-03, 6.51560047e-03, 5.63512264e-03, 4.84296166e-03,
                                         4.14793478e-03, 3.56267297e-03, 3.03480656e-03, 2.59406730e-03,
                                         2.19519042e-03, 1.85454286e-03, 1.58333238e-03, 1.33002321e-03,
                                         1.11716435e-03, 9.35360376e-04, 7.87780158e-04, 6.58601602e-04,
                                         5.48147154e-04, 4.58151351e-04, 3.85878963e-04, 3.21623249e-04,
                                         2.66129174e-04, 2.21364768e-04, 1.80736566e-04, 1.52350196e-04])
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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2, cm_prior='normal',
                    serial_interval_mean=SI_ALPHA / SI_BETA, serial_interval_sigma=np.sqrt(SI_ALPHA / SI_BETA ** 2),
                    conf_noise=None, deaths_noise=None
                    ):
        with self.model:
            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'icl':
                self.CM_Alpha_t = pm.Gamma("CM_Alpha_t", 1 / 6, 1, shape=(self.nCMs,))
                self.CM_Alpha = pm.Deterministic("CM_Alpha", self.CM_Alpha_t - np.log(1.05) / 6)

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic("RegionR", R_hyperprior_mean + self.RegionLogR_noise * self.HyperRVar)

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
                T.reshape(pm.math.log(self.RegionR), (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones((self.nORs, self.nDs))),
                                           plot_trace=False
                                           )

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, self.DailyGrowthNoise, shape=(self.nORs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, self.DailyGrowthNoise,
                                               shape=(self.nORs, self.nDs))

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth + self.GrowthCasesNoise)
            self.GrowthDeaths = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth + self.GrowthDeathsNoise)

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nORs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

            expected_cases = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbCases, newshape=(1, self.DelayProbCases.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed conf noise
            self.OutputNoiseScale = pm.HalfNormal('OutputNoiseScale', 1)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                pm.math.log(self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active]),
                self.OutputNoiseScale,
                shape=(len(self.all_observed_active),),
                observed=pm.math.log(self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active])
            )

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nORs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed deaths noise
            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                pm.math.log(self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]),
                self.OutputNoiseScale,
                shape=(len(self.all_observed_deaths),),
                observed=pm.math.log(self.d.NewDeaths.data[:, self.CMDelayCut:].reshape((self.nORs * self.nDs,))[
                                         self.all_observed_deaths])
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

    def plot_subset_region_predictions(self, region_indxs, plot_style, n_rows=3, fig_height=11, save_fig=True,
                                       output_dir="./out"):
        assert self.trace is not None

        for i, country_indx in enumerate(region_indxs):

            region = self.d.Rs[country_indx]

            if i % n_rows == 0:
                plt.figure(figsize=(10, fig_height), dpi=300)

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 1)

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

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            ids = self.trace.InfectedDeaths[:, country_indx, :]
            try:
                ed_output = dist.random()
            except:
                print("hi?")
                print(region)
                ed_output = np.ones_like(ids) * 10 ** -5
                ids = np.ones_like(ids) * 10 ** -5

            # if np.isnan(self.d.Deaths.data[country_indx, -1]):
            #     ed_output = np.ones_like(ids) * 10 ** -5
            #     ids = np.ones_like(ids) * 10 ** -5

            means_id, lu_id, up_id, err_id = produce_CIs(
                ids
            )

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

            plt.scatter(
                self.ObservedDaysIndx,
                newcases.data[self.ObservedDaysIndx],
                label="New Cases (Smoothed)",
                marker="o",
                s=10,
                color="tab:blue",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.scatter(
                self.ObservedDaysIndx,
                deaths.data[self.ObservedDaysIndx],
                label="New Deaths (Smoothed)",
                marker="o",
                s=10,
                color="tab:red",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 2)

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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 3)
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

            if i % n_rows == (n_rows - 1) or country_indx == len(self.d.Rs) - 1:
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
                        f"Fits{((country_indx + 1) / 5):.1f}"
                    )


class CMCombined_FinalLegacyAltSize(BaseCMModelLegacy):
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

        self.DelayProbDeaths = np.array([0.00000000e+00, 2.24600347e-06, 3.90382088e-05, 2.34307085e-04,
                                         7.83555003e-04, 1.91221622e-03, 3.78718437e-03, 6.45923913e-03,
                                         9.94265709e-03, 1.40610714e-02, 1.86527920e-02, 2.34311421e-02,
                                         2.81965055e-02, 3.27668001e-02, 3.68031574e-02, 4.03026198e-02,
                                         4.30521951e-02, 4.50637136e-02, 4.63315047e-02, 4.68794406e-02,
                                         4.67334059e-02, 4.59561441e-02, 4.47164503e-02, 4.29327455e-02,
                                         4.08614522e-02, 3.85082076e-02, 3.60294203e-02, 3.34601703e-02,
                                         3.08064505e-02, 2.81766028e-02, 2.56165924e-02, 2.31354369e-02,
                                         2.07837267e-02, 1.86074383e-02, 1.65505661e-02, 1.46527043e-02,
                                         1.29409383e-02, 1.13695920e-02, 9.93233881e-03, 8.66063386e-03,
                                         7.53805464e-03, 6.51560047e-03, 5.63512264e-03, 4.84296166e-03,
                                         4.14793478e-03, 3.56267297e-03, 3.03480656e-03, 2.59406730e-03,
                                         2.19519042e-03, 1.85454286e-03, 1.58333238e-03, 1.33002321e-03,
                                         1.11716435e-03, 9.35360376e-04, 7.87780158e-04, 6.58601602e-04,
                                         5.48147154e-04, 4.58151351e-04, 3.85878963e-04, 3.21623249e-04,
                                         2.66129174e-04, 2.21364768e-04, 1.80736566e-04, 1.52350196e-04])
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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2, cm_prior='normal',
                    serial_interval_mean=SI_ALPHA / SI_BETA, serial_interval_sigma=np.sqrt(SI_ALPHA / SI_BETA ** 2),
                    conf_noise=None, deaths_noise=None
                    ):
        with self.model:
            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'icl':
                self.CM_Alpha_t = pm.Gamma("CM_Alpha_t", 1 / 6, 1, shape=(self.nCMs,))
                self.CM_Alpha = pm.Deterministic("CM_Alpha", self.CM_Alpha_t - np.log(1.05) / 6)

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic("RegionR", R_hyperprior_mean + self.RegionLogR_noise * self.HyperRVar)

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
                T.reshape(pm.math.log(self.RegionR), (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.ExpectedGrowth = self.Det("ExpectedGrowth",
                                           si_beta * (pm.math.exp(
                                               self.ExpectedLogR / si_alpha) - T.ones((self.nORs, self.nDs))),
                                           plot_trace=False
                                           )

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, self.DailyGrowthNoise, shape=(self.nORs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, self.DailyGrowthNoise,
                                               shape=(self.nORs, self.nDs))

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth + self.GrowthCasesNoise)
            self.GrowthDeaths = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth + self.GrowthDeathsNoise)

            self.Tau1 = pm.Exponential("Tau1", 1 / 0.03)
            self.InitialSizeCases = pm.Exponential("InitialSizeCases", self.Tau1, shape=(self.nORs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases",
                                                  self.InitialSizeCases * pm.math.exp(self.GrowthCases.cumsum(axis=1)))

            expected_cases = C.conv2d(
                self.InfectedCases,
                np.reshape(self.DelayProbCases, newshape=(1, self.DelayProbCases.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed conf noise
            self.OutputNoiseScale = pm.HalfNormal('OutputNoiseScale', 1)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                pm.math.log(self.ExpectedCases.reshape((self.nORs * self.nDs,))[self.all_observed_active]),
                self.OutputNoiseScale,
                shape=(len(self.all_observed_active),),
                observed=pm.math.log(self.d.NewCases.data.reshape((self.nORs * self.nDs,))[self.all_observed_active])
            )

            self.Tau2 = pm.Exponential("Tau2", 1 / 0.03)
            self.InitialSizeDeaths = pm.Exponential("InitialSizeDeaths", self.Tau2, shape=(self.nORs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", self.InitialSizeDeaths * pm.math.exp(
                self.GrowthDeaths.cumsum(axis=1)))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            # can use learned or fixed deaths noise
            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                pm.math.log(self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]),
                self.OutputNoiseScale,
                shape=(len(self.all_observed_deaths),),
                observed=pm.math.log(self.d.NewDeaths.data[:, self.CMDelayCut:].reshape((self.nORs * self.nDs,))[
                                         self.all_observed_deaths])
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

    def plot_subset_region_predictions(self, region_indxs, plot_style, n_rows=3, fig_height=11, save_fig=True,
                                       output_dir="./out"):
        assert self.trace is not None

        for i, country_indx in enumerate(region_indxs):

            region = self.d.Rs[country_indx]

            if i % n_rows == 0:
                plt.figure(figsize=(10, fig_height), dpi=300)

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 1)

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

            ed = self.trace.ExpectedDeaths[:, country_indx, :]
            nS, nDs = ed.shape
            dist = pm.NegativeBinomial.dist(mu=ed + 1e-3, alpha=np.repeat(np.array([self.trace.Phi_1]), nDs, axis=0).T)

            ids = self.trace.InfectedDeaths[:, country_indx, :]
            try:
                ed_output = dist.random()
            except:
                print("hi?")
                print(region)
                ed_output = np.ones_like(ids) * 10 ** -5
                ids = np.ones_like(ids) * 10 ** -5

            # if np.isnan(self.d.Deaths.data[country_indx, -1]):
            #     ed_output = np.ones_like(ids) * 10 ** -5
            #     ids = np.ones_like(ids) * 10 ** -5

            means_id, lu_id, up_id, err_id = produce_CIs(
                ids
            )

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

            plt.scatter(
                self.ObservedDaysIndx,
                newcases.data[self.ObservedDaysIndx],
                label="New Cases (Smoothed)",
                marker="o",
                s=10,
                color="tab:blue",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.scatter(
                self.ObservedDaysIndx,
                deaths.data[self.ObservedDaysIndx],
                label="New Deaths (Smoothed)",
                marker="o",
                s=10,
                color="tab:red",
                alpha=0.9,
                zorder=4,
                facecolor="white"
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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 2)

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

            plt.subplot(n_rows, 3, 3 * (i % n_rows) + 3)
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

            if i % n_rows == (n_rows - 1) or country_indx == len(self.d.Rs) - 1:
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
                        f"Fits{((country_indx + 1) / 5):.1f}"
                    )
