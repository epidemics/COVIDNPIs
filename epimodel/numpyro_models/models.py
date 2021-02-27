"""
:code:`models.py`

Contains a variety of models of NPI effectiveness, all subclassed from BaseCMModel. 
"""
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import deterministic, plate, sample
from numpyro.distributions.continuous import HalfNormal

from ..pymc3_models.epi_params import EpidemiologicalParameters
from .base_model import BaseCMModel


class ComplexDifferentEffectsModel(BaseCMModel):
    """
    Complex Different Effects Model

    Note: this is the default model used by the paper!

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name="", model=None):
        super().__init__(data, cm_plot_style, name, model)
        self.country_specific_effects = True

    def build_model(
        self,
        R_prior_mean=3.28,
        cm_prior_scale=10,
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        alpha_noise_scale_prior="half-t",
        alpha_noise_scale=0.04,
        deaths_delay_mean_mean=21,
        deaths_delay_mean_sd=1,
        deaths_delay_disp_mean=9,
        deaths_delay_disp_sd=1,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        deaths_truncation=48,
        cases_truncation=32,
        growth_noise_scale="prior",
        **kwargs,
    ):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale.
        :param cm_prior: mean NPI effectiveness across countries prior type
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param alpha_noise_scale_prior: prior type placed over sigma_i. either 'half-normal' or 'half-t'
        :param alpha_noise_scale: scale of sigma_i prior.
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        with self.model:
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            if alpha_noise_scale_prior == "half-normal":
                self.CMAlphaScales = pm.HalfNormal(
                    "CMAlphaScales", sigma=alpha_noise_scale, shape=(self.nCMs)
                )
            elif alpha_noise_scale_prior == "half-t":
                self.CMAlphaScales = pm.HalfStudentT(
                    "CMAlphaScales", nu=3, sigma=alpha_noise_scale, shape=(self.nCMs)
                )

            self.AllCMAlphaNoise = pm.Normal(
                "AllCMAlphaNoise", 0, 1, shape=(self.nRs, self.nCMs)
            )
            self.AllCMAlpha = pm.Deterministic(
                "AllCMAlpha",
                T.reshape(self.CM_Alpha, (1, self.nCMs)).repeat(self.nRs, axis=0)
                + self.CMAlphaScales.reshape((1, self.nCMs)) * self.AllCMAlphaNoise,
            )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=0.5)

            self.RegionR_noise = pm.Normal("RegionR_noise", 0, 1, shape=(self.nRs),)
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise * self.HyperRVar
            )

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            active_cm_reduction = (
                T.reshape(self.AllCMAlpha, (self.nRs, self.nCMs, 1)) * self.ActiveCMs
            )
            growth_reduction = T.sum(active_cm_reduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction,
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (
                pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs))
            )

            if growth_noise_scale == "prior":
                self.GrowthNoiseScale = pm.HalfStudentT(
                    "GrowthNoiseScale", nu=3, sigma=0.15
                )
            elif growth_noise_scale == "fixed":
                self.GrowthNoiseScale = 0.205
                self.GrowthNoiseScale
            elif growth_noise_scale == "indep":
                self.GrowthNoiseScaleCases = pm.HalfStudentT(
                    "GrowthNoiseCases", nu=3, sigma=0.15
                )
                self.GrowthNoiseScaleDeaths = pm.HalfStudentT(
                    "GrowthNoiseDeaths", nu=3, sigma=0.15
                )

            # exclude 40 days of noise, slight increase in runtime.
            self.GrowthCasesNoise = pm.Normal(
                "GrowthCasesNoise", 0, 1, shape=(self.nRs, self.nDs - 40)
            )
            self.GrowthDeathsNoise = pm.Normal(
                "GrowthDeathsNoise", 0, 1, shape=(self.nRs, self.nDs - 40)
            )

            self.GrowthCases = T.inc_subtensor(
                self.ExpectedGrowth[:, 30:-10],
                self.GrowthNoiseScale * self.GrowthCasesNoise,
            )
            self.GrowthDeaths = T.inc_subtensor(
                self.ExpectedGrowth[:, 30:-10],
                self.GrowthNoiseScale * self.GrowthDeathsNoise,
            )

            self.PsiCases = pm.HalfNormal("PsiCases", 5.0)
            self.PsiDeaths = pm.HalfNormal("PsiDeaths", 5.0)

            self.InitialSizeCases_log = pm.Normal(
                "InitialSizeCases_log", 0, 50, shape=(self.nRs,)
            )
            self.InfectedCases_log = pm.Deterministic(
                "InfectedCases_log",
                T.reshape(self.InitialSizeCases_log, (self.nRs, 1))
                + self.GrowthCases.cumsum(axis=1),
            )
            self.InfectedCases = pm.Deterministic(
                "InfectedCases", pm.math.exp(self.InfectedCases_log)
            )

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_cases.reshape((self.nRs, self.nDs))
            )

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

            self.InitialSizeDeaths_log = pm.Normal(
                "InitialSizeDeaths_log", 0, 50, shape=(self.nRs,)
            )
            self.InfectedDeaths_log = pm.Deterministic(
                "InfectedDeaths_log",
                T.reshape(self.InitialSizeDeaths_log, (self.nRs, 1))
                + self.GrowthDeaths.cumsum(axis=1),
            )
            self.InfectedDeaths = pm.Deterministic(
                "InfectedDeaths", pm.math.exp(self.InfectedDeaths_log)
            )

            self.DeathsDelayMean = pm.Normal(
                "DeathsDelayMean", deaths_delay_mean_mean, deaths_delay_mean_sd
            )
            self.DeathsDelayDisp = pm.Normal(
                "DeathsDelayDisp", deaths_delay_disp_mean, deaths_delay_disp_sd
            )
            deaths_delay_dist = pm.NegativeBinomial.dist(
                mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp
            )
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths, fatality_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedDeaths = pm.Deterministic(
                "ExpectedDeaths", expected_deaths.reshape((self.nRs, self.nDs))
            )

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
            )

