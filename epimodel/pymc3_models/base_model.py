"""
base_model file, which contains:

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

fp2 = FontProperties(fname=r"../../../fonts/Font Awesome 5 Free-Solid-900.otf")
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

    :param ax:
    :param ActiveCMs:
    :param country_indx:
    :param min_x:
    :param max_x:
    :param days:
    :param plot_style:
    :return:
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
    def __init__(
            self, data, cm_plot_style, name="", model=None
    ):
        """

        :param data:
        :param cm_plot_style:
        :param name:
        :param model:
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

    @property
    def nRs(self):
        """

        :return:
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return:
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return:
        """
        return len(self.d.CMs)

    def build_npi_prior(self, prior_type, prior_scale=None):
        with self.model:
            if prior_type == 'normal':
                self.CM_Alpha = pm.Normal("CM_Alpha", 0, prior_scale, shape=(self.nCMs,))

            elif prior_type == 'half_normal':
                self.CM_Alpha = pm.HalfNormal("CM_Alpha", prior_scale, shape=(self.nCMs,))

            elif prior_type == 'icl':
                self.CM_Alpha_t = pm.Gamma("CM_Alpha_t", 1 / self.nCMs, 1, shape=(self.nCMs,))
                self.CM_Alpha = pm.Deterministic("CM_Alpha", self.CM_Alpha_t - np.log(1.05) / self.nCMs)

            elif prior_type == 'skewed':
                self.CM_Alpha = AsymmetricLaplace(scale=prior_scale, symmetry=0.5)

    def plot_effect(self):
        """

        :return:
        """
        assert self.trace is not None
        plt.figure(figsize=(4, 3), dpi=300)

        m, li, ui, lq, uq = produce_CIs()
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

        plt.xlim([-100, 100])
        xtick_vals = np.arange(-100, 150, 50)
        xtick_str = [f"{x:.0f}%" for x in xtick_vals]
        plt.ylim([-(N_cms - 0.5), 0.5])

        plt.yticks(
            -np.arange(len(self.d.CMs)),
            [f"{f}" for f in self.d.CMs]
        )

        x_min, x_max = plt.xlim()

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
