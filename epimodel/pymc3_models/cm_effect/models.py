import copy
import logging
import os
from datetime import datetime

import seaborn as sns

import numpy as np
import scipy.stats
import pymc3 as pm
import theano
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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_sigma=0.2, cm_prior='normal',
                    serial_interval_mean=SI_ALPHA / SI_BETA, conf_noise=None, deaths_noise=None
                    ):
        with self.model:
            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

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
                    observed=self.d.NewDeaths.data.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]
                )

            self.Det(
                "Z2D",
                self.ObservedDeaths - self.ExpectedDeaths.reshape((self.nORs * self.nDs,))[self.all_observed_deaths]
            )

class CMCombined_Final_V3(BaseCMModel):
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
        self.DailyGrowthNoise = 0.3

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

            self.ExpectedLogRCases = pm.Normal(
                "ExpectedLogRCases",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs)
            )

            self.ExpectedLogRDeaths = pm.Normal(
                "ExpectedLogRDeaths",
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                self.DailyGrowthNoise,
                shape=(self.nORs, self.nDs)
            )

            serial_interval_sigma = np.sqrt(SI_ALPHA / SI_BETA ** 2)
            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2

            self.GrowthCases = self.Det("GrowthCases",
                                        si_beta * (pm.math.exp(
                                            self.ExpectedLogRCases / si_alpha) - T.ones_like(
                                            self.ExpectedLogRCases)),
                                        plot_trace=False
                                        )

            self.GrowthDeaths = self.Det("GrowthDeaths",
                                         si_beta * (pm.math.exp(
                                             self.ExpectedLogRDeaths / si_alpha) - T.ones_like(
                                             self.ExpectedLogRDeaths)),
                                         plot_trace=False
                                         )

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

class CMCombined_Additive(BaseCMModel):
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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_conc=1,
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:
            self.AllBeta = pm.Dirichlet("AllBeta", cm_prior_conc * np.ones((self.nCMs + 1)), shape=(self.nCMs + 1,))
            self.CM_Beta = pm.Deterministic("CM_Beta", self.AllBeta[1:])
            self.Beta_hat = pm.Deterministic("Beta_hat", self.AllBeta[0])
            self.CMReduction = pm.Deterministic("CMReduction", self.CM_Beta)

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
                    T.reshape(self.CM_Beta, (1, self.nCMs, 1))
                    * (T.ones_like(self.ActiveCMs[self.OR_indxs, :, :]) - self.ActiveCMs[self.OR_indxs, :, :])
            )

            self.Det(
                "GrowthReduction", T.sum(self.ActiveCMReduction, axis=1) + self.Beta_hat, plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                "ExpectedLogR",
                T.log(T.exp(T.reshape(self.RegionLogR, (self.nORs, 1))) * self.GrowthReduction),
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

    def plot_effect(self, save_fig=True, output_dir="./out", x_min=-100, x_max=100):
        assert self.trace is not None
        fig = plt.figure(figsize=(9, 3), dpi=300)
        plt.subplot(121)
        self.d.coactivation_plot(self.cm_plot_style, newfig=False)
        plt.subplot(122)

        means = 100 * (np.mean(self.trace["AllBeta"], axis=0))
        li = 100 * (np.percentile(self.trace["AllBeta"], 5, axis=0))
        ui = 100 * (np.percentile(self.trace["AllBeta"], 95, axis=0))
        lq = 100 * (np.percentile(self.trace["AllBeta"], 25, axis=0))
        uq = 100 * (np.percentile(self.trace["AllBeta"], 75, axis=0))

        N_cms = means.size

        plt.plot([0, 0], [1, -(N_cms)], "--r", linewidth=0.5)
        y_vals = -1 * np.arange(N_cms)
        plt.scatter(means, y_vals, marker="|", color="k")
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.5)

        xtick_vals = np.arange(-100, 150, 50)
        xtick_str = [f"{x:.0f}%" for x in xtick_vals]
        plt.ylim([-(N_cms - 0.5), 0.5])

        ylabels = ["base"]
        ylabels.extend(self.d.CMs)

        plt.yticks(
            -np.arange(len(self.d.CMs) + 1),
            [f"{f}" for f in ylabels]
        )

        # ax = plt.gca()
        # x_min, x_max = plt.xlim()
        # x_r = x_max - x_min
        # print(x_r)
        # for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
        #     ticklabel.set_color(self.cm_plot_style[i][1])
        #     plt.text(x_min - 0.13 * x_r, tickloc, self.cm_plot_style[i][0], horizontalalignment='center',
        #              verticalalignment='center',
        #              fontproperties=fp2, fontsize=10, color=self.cm_plot_style[i][1])

        plt.xticks(xtick_vals, xtick_str, fontsize=6)
        plt.xlim([-10, 75])
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


class CMCombined_Final_DifEffects(BaseCMModel):
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
        self.RegionVariationNoise = 0.05

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
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:
            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            self.AllCMAlpha = pm.Normal("AllCMAlpha",
                                        T.reshape(self.CM_Alpha, (1, self.nCMs)).repeat(self.nORs, axis=0),
                                        self.RegionVariationNoise,
                                        shape=(self.nORs, self.nCMs)
                                        )

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
                    T.reshape(self.AllCMAlpha, (self.nORs, self.nCMs, 1))
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

class CMCombined_ICL_NoNoise(BaseCMModel):
    def __init__(
            self, data, name="", model=None, cm_plot_style=None
    ):
        super().__init__(data, cm_plot_style=cm_plot_style, name=name, model=model)

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

        self.ObservedDaysIndx = np.arange(self.CMDelayCut, len(self.d.Ds))
        self.OR_indxs = np.arange(len(self.d.Rs))
        self.nORs = self.nRs
        self.nODs = len(self.ObservedDaysIndx)
        self.ORs = copy.deepcopy(self.d.Rs)

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
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:

            serial_interval_sigma = np.sqrt(SI_ALPHA / SI_BETA ** 2)
            si_beta = serial_interval_mean / serial_interval_sigma ** 2
            si_alpha = serial_interval_mean ** 2 / serial_interval_sigma ** 2
            x = np.arange(len(self.SI)) * 1.
            if serial_interval_mean < 5:  # to avoid inf first value for small means
                x[0] = 0.001
            self.SI = scipy.stats.gamma.pdf(x, si_alpha, scale=1 / si_beta)
            self.SI_rev = self.SI[::-1].reshape((1, self.SI.size))

            if cm_prior == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, cm_prior_sigma, shape=(self.nCMs,))

            if cm_prior == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", cm_prior_sigma, shape=(self.nCMs,))

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

            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nORs,))

            # # conv padding
            filter_size = self.SI_rev.size
            conv_padding = 7

            infected_deaths = T.zeros((self.nORs, self.nDs + self.SI_rev.size))
            infected_deaths = T.set_subtensor(infected_deaths[:, (filter_size - conv_padding):filter_size],
                                              pm.math.exp(self.InitialSizeDeaths_log.reshape((self.nORs, 1)).repeat(
                                                  conv_padding, axis=1)))

            # R is a lognorm
            R_deaths = pm.math.exp(self.ExpectedLogR)

            for d in range(self.nDs):
                val_d = pm.math.sum(
                    R_deaths[:, d].reshape((self.nORs, 1)) * infected_deaths[:, d:d + filter_size] * self.SI_rev,
                    axis=1)
                infected_deaths = T.set_subtensor(infected_deaths[:, d + filter_size], val_d)

            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", infected_deaths[:, filter_size:])

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                np.reshape(self.DelayProbDeaths, newshape=(1, self.DelayProbDeaths.size)),
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nORs, self.nDs)))

            self.Phi = pm.HalfNormal("Phi", 5)

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
