"""
base_model file

This file contains contains:
- utility functions used often with models
- BaseCMModel class, which other classes inherit from
"""

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
from matplotlib.font_manager import FontProperties
from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace
from pymc3 import Model

fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")
sns.set_style("ticks")


def produce_CIs(array):
    """
    Produce 95%, 50% Confidence intervals from a Numpy array, taking CIs using the 0th axis.

    :param array: Numpy array from which to compute CIs.
    :return: (median, 2.5 percentile, 97.5 percentile, 25th percentile, 75th percentile) tuple.
    """
    m = np.median(array, axis=0)
    li = np.percentile(array, 2.5, axis=0)
    ui = np.percentile(array, 97.5, axis=0)
    uq = np.percentile(array, 75, axis=0)
    lq = np.percentile(array, 25, axis=0)
    return m, li, ui, lq, uq


def add_cms_to_plot(ax, ActiveCMs, country_indx, min_x, max_x, days, plot_style):
    """
    Plotter helper.

    This takes a plot and adds NPI logos on them.

    :param ax: axis to draw
    :param ActiveCMs: Standard ActiveCMs numpy array
    :param country_indx: Country to pull CM data from
    :param min_x: x limit - left
    :param max_x: x limit - right
    :param days: days to plot
    :param plot_style: NPI Plot style
    """
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
    """
    BaseCMModel Class.

    Other NPI models inherit from this class
    """

    def __init__(
            self, data, cm_plot_style=None, name="", model=None
    ):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None

        if cm_plot_style is not None:
            self.cm_plot_style = cm_plot_style
        else:
            # for pretty symbols
            self.cm_plot_style = [
                ("\uf963", "black"),  # mask
                ("\uf0c0", "darkgrey"),  # ppl
                ("\uf0c0", "dimgrey"),  # ppl
                ("\uf0c0", "black"),  # ppl
                ("\uf07a", "tab:orange"),  # shop 1
                ("\uf07a", "tab:red"),  # shop2
                ("\uf549", "black"),  # school
                ("\uf19d", "black"),  # university
                ("\uf965", "black")  # home
            ]

        # don't observe deaths before ~22nd feb
        self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        observed_active = []
        observed_deaths = []
        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if self.d.NewCases.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Confirmed.data[r, d]):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

                if self.d.NewDeaths.mask[r, d] == False and d > self.CMDelayCut and not np.isnan(
                        self.d.Deaths.data[r, d]):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        self.all_observed_active = np.array(observed_active)
        self.all_observed_deaths = np.array(observed_deaths)

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

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_npi_prior(self, prior_type, prior_scale=None):
        """
        Build NPI Effectiveness Prior.

        There are four options:
        Normal - alpha_i is normally distributed with given scale.
        Half Normal - alpha_i is half normally distributed with given scale (i.e., restricted to positive effect)
        Flaxman/ICL prior - As used in Flaxman et al.
        Skewed - asymmetric laplace prior.

        :param prior_type: Choose one of normal, half-normal, icl or skewed.
        :param prior_scale: Prior scale. Relevant for normal, half-normal and skewed distributions.
        """
        with self.model:
            if prior_type == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, prior_scale, shape=(self.nCMs,))

            elif prior_type == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", prior_scale, shape=(self.nCMs,))

            elif prior_type == 'icl':
                self.CM_Alpha_t = pm.Gamma("CM_Alpha_t", 1 / self.nCMs, 1, shape=(self.nCMs,))
                self.CM_Alpha = pm.Deterministic("CM_Alpha", self.CM_Alpha_t - np.log(1.05) / self.nCMs)

            elif prior_type == 'skewed':
                self.CM_Alpha = AsymmetricLaplace('CM_Alpha', scale=prior_scale, symmetry=0.5, shape=(self.nCMs,))

    def plot_effect(self):
        """
        If model.trace has been set, plot the NPI effectiveness estimates.
        """
        assert self.trace is not None
        plt.figure(figsize=(4, 3), dpi=300)

        means, li, ui, lq, uq = produce_CIs(100 * (1 - np.mean(self.trace["CMReduction"])))

        N_cms = means.size

        plt.plot([0, 0], [1, -(N_cms)], "--r", linewidth=0.5)
        y_vals = -1 * np.arange(N_cms)
        plt.scatter(means, y_vals, marker="|", color="k")
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm], y_vals[cm]], "k", alpha=0.5)

        plt.xlim([-100, 100])
        xtick_vals = np.arange(-100, 150, 50)
        xtick_str = [f"{x:.0f}%" for x in xtick_vals]
        plt.ylim([-(N_cms - 0.5), 0.5])

        plt.yticks(
            -np.arange(len(self.d.CMs)),
            [f"{f}" for f in self.d.CMs]
        )

        plt.xticks(xtick_vals, xtick_str, fontsize=6)
        plt.xlabel("Average Additional Reduction in $R$", fontsize=8)
        plt.tight_layout()

        fig = plt.figure(figsize=(7, 3), dpi=300)
        correlation = np.corrcoef(self.trace["CMReduction"], rowvar=False)
        plt.imshow(correlation, cmap="PuOr", vmin=-1, vmax=1)
        cbr = plt.colorbar()
        cbr.ax.tick_params(labelsize=6)
        plt.yticks(np.arange(N_cms), self.d.CMs, fontsize=6)
        plt.xticks(np.arange(N_cms), self.d.CMs, fontsize=6, rotation=90)
        plt.title("Posterior Correlation", fontsize=10)
        sns.despine()
