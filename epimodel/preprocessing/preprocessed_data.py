"""
Contains PreprocessedData Class definition.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
import seaborn as sns

sns.set_style('ticks')
fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")


class PreprocessedData(object):
    """
    PreprocessedData Class

    Class to hold data which is subsequently passed onto a PyMC3 model. Mostly a data wrapper, with some utility
    functions.
    """

    def __init__(self,
                 Active,
                 Confirmed,
                 ActiveCMs,
                 CMs,
                 Rs,
                 Ds,
                 Deaths,
                 NewDeaths,
                 NewCases,
                 RNames):
        """
        Constructor.

        Note: nRs is the number of regions. nDs is the number of days

        :param Active: Active Cases np.ndarray array. Shape is (nRs, nDs)
        :param Confirmed: Confirmed Cases np.ndarray array. Shape is (nRs, nDs)
        :param ActiveCMs: NPI Interventino data np.ndarray binary array. Shape is (nRs, nCMs, nDs)
        :param CMs: List containing intervention names
        :param Rs: List containing region names
        :param Ds: List containing days
        :param Deaths: Deaths np.ndarray array. Shape is (nRs, nDs)
        :param NewDeaths: Daily Deaths np.ndarray array. Shape is (nRs, nDs). Note: this is usually a masked array.
        :param NewCases: Daily Cases np.ndarray array. Shape is (nRs, nDs). Note: this is usually a masked array.
        :param RNames: Region names
        """
        super().__init__()
        self.Active = Active
        self.Confirmed = Confirmed
        self.Deaths = Deaths
        self.ActiveCMs = ActiveCMs
        self.Rs = Rs
        self.CMs = CMs
        self.Ds = Ds
        self.NewDeaths = NewDeaths
        self.NewCases = NewCases
        self.RNames = RNames

    def reduce_regions_from_index(self, reduced_regions_indx):
        """
        Reduce data to only pertain to region indices given. Occurs in place.

        e.g., if reduced_regions_indx = [0], the resulting data object will contain data about only the first region.

        :param reduced_regions_indx: region indices to retain.
        """
        self.Active = self.Active[reduced_regions_indx, :]
        self.Confirmed = self.Confirmed[reduced_regions_indx, :]
        self.Deaths = self.Deaths[reduced_regions_indx, :]
        self.NewDeaths = self.NewDeaths[reduced_regions_indx, :]
        self.NewCases = self.NewCases[reduced_regions_indx, :]
        self.ActiveCMs = self.ActiveCMs[reduced_regions_indx, :, :]

    def remove_regions_min_deaths(self, min_num_deaths=100):
        """
        Remove regions which have fewer than min_num_deaths at the end of the considered time period. Occurs in place.

        :param min_num_deaths: Minimum number of (total) deaths.
        """
        reduced_regions = []
        reduced_regions_indx = []
        for indx, r in enumerate(self.Rs):
            if self.Deaths.data[indx, -1] < min_num_deaths:
                print(f"Region {r} removed since it has {self.Deaths[indx, -1]} deaths on the last day")
            elif np.isnan(self.Deaths.data[indx, -1]):
                print(f"Region {r} removed since it has {self.Deaths[indx, -1]} deaths on the last day")
            else:
                reduced_regions.append(r)
                reduced_regions_indx.append(indx)

        self.Rs = reduced_regions
        self.reduce_regions_from_index(reduced_regions_indx)

    def remove_regions_from_codes(self, regions_to_remove):
        """
        Remove region codes corresponding to regions in regions_to_remove. Occurs in place.

        :param regions_to_remove: Region codes, corresponding to regions to remove.
        """
        reduced_regions = []
        reduced_regions_indx = []
        for indx, r in enumerate(self.Rs):
            if r in regions_to_remove:
                pass
            else:
                reduced_regions_indx.append(indx)
                reduced_regions.append(r)

        self.Rs = reduced_regions
        _, nCMs, nDs = self.ActiveCMs.shape
        self.reduce_regions_from_index(reduced_regions_indx)

    def conditional_activation_plot(self, cm_plot_style, newfig=True, skip_yticks=False):
        """
        Draw conditional-activation plot.

        :param cm_plot_style: Countermeasure plot style array.
        :param newfig: boolean, whether to create plot in a new figure
        :param skip_yticks: boolean, whether to draw yticks.
        """
        if newfig:
            plt.figure(figsize=(2, 3), dpi=300)

        nRs, nCMs, nDs = self.ActiveCMs.shape
        plt.title("Frequency $i$ Active Given $j$ Active", fontsize=8)
        ax = plt.gca()
        mat = np.zeros((nCMs, nCMs))
        for cm in range(nCMs):
            mask = self.ActiveCMs[:, cm, :] * (self.NewDeaths.mask == False)
            for cm2 in range(nCMs):
                mat[cm, cm2] = np.sum(mask * self.ActiveCMs[:, cm2, :]) / np.sum(mask)
        im = plt.imshow(mat * 100, vmin=25, vmax=100, cmap='inferno', aspect="auto")
        ax.tick_params(axis="both", which="major", labelsize=8)

        for i in range(nCMs):
            for j in range(nCMs):
                if mat[i, j] < 0.75:
                    plt.text(j, i, f'{int(100 * mat[i, j]):d}%', fontsize=3.5, ha='center', va='center', color='white',
                             rotation=45)
                else:
                    plt.text(j, i, f'{int(100 * mat[i, j]):d}%', fontsize=3.5, ha='center', va='center', color=[0.4627010031973002, 0.2693410356621817, 0.46634810758714684],
                             rotation=45)

        plt.xticks(
            np.arange(len(self.CMs)),
            [f"{cm_plot_style[i][0]}" for i, f in enumerate(self.CMs)],
            fontproperties=fp2,
        )

        for i, ticklabel in enumerate(ax.get_xticklabels()):
            ticklabel.set_color(cm_plot_style[i][1])

        plt.yticks(
            np.arange(len(self.CMs)),
            [f"{f}     " if not skip_yticks else "    " for f in self.CMs]
        )

        plt.xlabel("$i$", fontsize=8)
        plt.ylabel("$j$", fontsize=8)

        x_min, x_max = plt.xlim()
        x_r = x_max - x_min
        for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
            ticklabel.set_color(cm_plot_style[i][1])
            plt.text(-0.16 * x_r, tickloc, cm_plot_style[i][0], horizontalalignment='center',
                     verticalalignment='center',
                     fontproperties=fp2, fontsize=7, color=cm_plot_style[i][1])

        plt.xticks(fontsize=7)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbr = plt.colorbar(im, cax=cax, format=PercentFormatter())
        # ax = plt.gca()
        # ax.tick_params(axis="both", which="major", labelsize=6)
        # cbr.set_ticks([25, 50, 75, 100])

    def cumulative_days_plot(self, cm_plot_style, newfig=True, skip_yticks=False):
        """
        Draw cumulative days plot.

        :param cm_plot_style: Countermeasure plot style array.
        :param newfig: boolean, whether to create plot in a new figure
        :param skip_yticks: boolean, whether to draw yticks.
        """
        if newfig:
            plt.figure(figsize=(3, 3), dpi=300)

        nRs, nCMs, nDs = self.ActiveCMs.shape

        ax = plt.gca()
        mask = np.reshape((self.NewDeaths.mask == False), (nRs, 1, nDs))
        days_active = np.sum(np.sum(self.ActiveCMs * np.repeat(mask, nCMs, axis=1), axis=0), axis=1)
        plt.barh(-np.arange(nCMs), days_active, color=[0.7522446028276593, 0.5089037847613617, 0.6733963201089419])

        plt.yticks(
            -np.arange(len(self.CMs)),
            [f"{f}     " if not skip_yticks else "    " for f in self.CMs]
        )

        x_min, x_max = plt.xlim()
        x_r = x_max - x_min
        for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
            ticklabel.set_color(cm_plot_style[i][1])
            plt.text(-0.09 * x_r, tickloc, cm_plot_style[i][0], horizontalalignment='center',
                     verticalalignment='center',
                     fontproperties=fp2, fontsize=7, color=cm_plot_style[i][1])

        plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], fontsize=6)
        # ax.tick_params(axis="both", which="major", labelsize=10)
        plt.title("Total Days Active", fontsize=8)
        plt.xlabel("Days", fontsize=8)
        plt.ylim([-len(self.CMs) + 0.5, 0.5])

    def summary_plot(self, cm_plot_style):
        """
        Draw summary plot.

        This includes both the cumulative days plot, and the conditional activation plot.

        :param cm_plot_style: Countermeasure plot style array.
        """
        plt.figure(figsize=(10, 3), dpi=300)
        plt.subplot(1, 2, 1)
        self.coactivation_plot(cm_plot_style, False)
        plt.subplot(1, 2, 2)
        self.cumulative_days_plot(cm_plot_style, False)
        plt.tight_layout()
        plt.savefig("FigureCA.pdf", bbox_inches='tight')
        # sns.despine()

    def mask_reopenings(self, d_min=90, n_extra=0, print_out=True):
        """
        Mask reopenings.

        This finds dates NPIs reactivate, then mask forwards, giving 3 days for cases and 12 days for deaths. 

        :param d_min: day after which to mask reopening.
        :param n_extra: int, number of extra days to mask
        """
        total_cms = self.ActiveCMs
        diff_cms = np.zeros_like(total_cms)
        diff_cms[:, :, 1:] = total_cms[:, :, 1:] - total_cms[:, :, :-1]
        rs, ds = np.nonzero(np.any(diff_cms < 0, axis=1))
        nnz = rs.size

        for nz_i in range(nnz):
            if (ds[nz_i] + 3) > d_min and ds[nz_i] + 3 < len(self.Ds):
                if print_out:
                    print(f"Masking {self.Rs[rs[nz_i]]} from {self.Ds[ds[nz_i] + 3]}")
                self.NewCases[rs[nz_i], ds[nz_i] + 3 - n_extra:].mask = True
                self.NewDeaths[rs[nz_i], ds[nz_i] + 12 - n_extra:].mask = True

    def mask_region_ends(self, n_days=20):
        """
        Mask the final n_days days across all countries.

        :param n_days: number of days to mask.
        """
        for rg in self.Rs:
            i = self.Rs.index(rg)
            self.Active.mask[i, -n_days:] = True
            self.Confirmed.mask[i, -n_days:] = True
            self.Deaths.mask[i, -n_days:] = True
            self.NewDeaths.mask[i, -n_days:] = True
            self.NewCases.mask[i, -n_days:] = True

    def mask_region(self, region, days=14):
        """
        Mask all but the first 14 days of cases and deaths for a specific region

        :param region: region code (2 digit EpidemicForecasting.org) code to mask
        :param days: Number of days to provide to the model
        """
        i = self.Rs.index(region)
        c_s = np.nonzero(np.cumsum(self.NewCases.data[i, :] > 0) == days + 1)[0][0]
        d_s = np.nonzero(np.cumsum(self.NewDeaths.data[i, :] > 0) == days + 1)[0]
        if len(d_s) > 0:
            d_s = d_s[0]
        else:
            d_s = len(self.Ds)

        self.Active.mask[i, c_s:] = True
        self.Confirmed.mask[i, c_s:] = True
        self.Deaths.mask[i, d_s:] = True
        self.NewDeaths.mask[i, d_s:] = True
        self.NewCases.mask[i, c_s:] = True

        return c_s, d_s

    def unmask_all(self):
        """
        Unmask all cases, deaths.
        """
        self.Active.mask = False
        self.Confirmed.mask = False
        self.Deaths.mask = False
        self.NewDeaths.mask = False
        self.NewCases.mask = False
