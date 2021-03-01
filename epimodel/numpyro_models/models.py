"""
:code:`models.py`

Contains a variety of models of NPI effectiveness, all subclassed from BaseCMModel. 
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import deterministic, plate, sample
from numpyro.distributions.continuous import HalfNormal

from ..pymc3_models.epi_params import EpidemiologicalParameters
from .base_model import BaseCMModel
from .distributions import NegativeBinomial, sample_half_student_t


class ComplexDifferentEffectsModel(BaseCMModel):
    """
    Complex Different Effects Model

    Note: this is the default model used by the paper!

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name=""):
        super().__init__(data, cm_plot_style, name)
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

        def run_model(data_ActiveCMs, NewCases_data, all_observed_active):
            nRs, nCMs, nDs = data_ActiveCMs.shape
            CM_Alpha = self.sample_npi_prior(cm_prior, cm_prior_scale)

            CMReduction = deterministic("CMReduction", jnp.exp((-1.0) * CM_Alpha))

            with plate("nCMs", nCMs):
                if alpha_noise_scale_prior == "half-normal":
                    CMAlphaScales = sample(
                        "CMAlphaScales", dist.HalfNormal(scale=alpha_noise_scale)
                    )
                elif alpha_noise_scale_prior == "half-t":
                    CMAlphaScales = sample_half_student_t(
                        "CMAlphaScales", nu=3, sigma=alpha_noise_scale
                    )

            AllCMAlphaNoise = sample(
                "AllCMAlphaNoise", dist.Normal(jnp.zeros((nRs, nCMs)), 1)
            )

            AllCMAlpha = deterministic(
                "AllCMAlpha",
                jnp.reshape(CM_Alpha, (1, nCMs)).repeat(nRs, axis=0)
                + CMAlphaScales.reshape((1, nCMs)) * AllCMAlphaNoise,
            )

            HyperRVar = sample("HyperRVar", dist.HalfNormal(scale=0.5))
            RegionR_noise = sample("RegionR_noise", dist.Normal(jnp.zeros(nRs), 1))
            RegionR = deterministic("RegionR", R_prior_mean + RegionR_noise * HyperRVar)

            ActiveCMs = deterministic("ActiveCMs", data_ActiveCMs)

            active_cm_reduction = (
                jnp.reshape(AllCMAlpha, (nRs, nCMs, 1)) * ActiveCMs
            )
            growth_reduction = jnp.sum(active_cm_reduction, axis=1)

            ExpectedLogR = deterministic(
                "ExpectedLogR",
                jnp.reshape(jnp.log(RegionR), (nRs, 1)) - growth_reduction,
            )

            # convert R into growth rates
            GI_mean = sample("GI_mean", dist.Normal(gi_mean_mean, gi_mean_sd))
            GI_sd = sample("GI_sd", dist.Normal(gi_sd_mean, gi_sd_sd))

            gi_beta = GI_mean / GI_sd ** 2
            gi_alpha = GI_mean ** 2 / GI_sd ** 2

            ExpectedGrowth = gi_beta * (
                jnp.exp(ExpectedLogR / gi_alpha) - jnp.ones((nRs, nDs))
            )

            if growth_noise_scale == "prior":
                GrowthNoiseScale = sample_half_student_t(
                    "GrowthNoiseScale", nu=3, sigma=0.15
                )
            elif growth_noise_scale == "fixed":
                GrowthNoiseScale = deterministic("GrowthNoiseScale", 0.205)
            elif growth_noise_scale == "indep":  ## NB: Now broken
                GrowthNoiseScaleCases = sample_half_student_t(
                    "GrowthNoiseCases", nu=3, sigma=0.15
                )
                GrowthNoiseScaleDeaths = sample_half_student_t(
                    "GrowthNoiseDeaths", nu=3, sigma=0.15
                )

            # exclude 40 days of noise, slight increase in runtime.
            GrowthCasesNoise = sample(
                "GrowthCasesNoise", dist.Normal(jnp.zeros((nRs, nDs - 40)), 1)
            )
            GrowthDeathsNoise = sample(
                "GrowthDeathsNoise",
                dist.Normal(jnp.zeros((nRs, nDs - 40)), 1),
            )

            GrowthCases = jnp.concatenate(
                (
                    (ExpectedGrowth[:, :30]),
                    ExpectedGrowth[:, 30:-10] + GrowthNoiseScale * GrowthCasesNoise,
                    ExpectedGrowth[:, -10:],
                ),
                axis=1,
            )

            GrowthDeaths = jnp.concatenate(
                (
                    (ExpectedGrowth[:, :30]),
                    ExpectedGrowth[:, 30:-10] + GrowthNoiseScale * GrowthDeathsNoise,
                    ExpectedGrowth[:, -10:],
                ),
                axis=1,
            )

            PsiCases = sample("PsiCases", dist.HalfNormal(5.0))
            PsiDeaths = sample("PsiDeaths", dist.HalfNormal(5.0))

            InitialSizeCases_log = sample(
                "InitialSizeCases_log", dist.Normal(jnp.zeros((nRs,)), 50)
            )
            InfectedCases_log = deterministic(
                "InfectedCases_log",
                jnp.reshape(InitialSizeCases_log, (nRs, 1))
                + GrowthCases.cumsum(axis=1),
            )
            InfectedCases = deterministic("InfectedCases", jnp.exp(InfectedCases_log))

            CasesDelayMean = sample(
                "CasesDelayMean",
                dist.Normal(cases_delay_mean_mean, cases_delay_mean_sd),
            )
            CasesDelayDisp = sample(
                "CasesDelayDisp",
                dist.Normal(cases_delay_disp_mean, cases_delay_disp_sd),
            )

            delay_prob = CasesDelayDisp / (CasesDelayMean + CasesDelayDisp)
            cases_delay_dist = NegativeBinomial(CasesDelayDisp, probs=delay_prob)
            bins = jnp.arange(0, cases_truncation)
            pmf = jnp.exp(cases_delay_dist.log_prob(bins))
            pmf = pmf / jnp.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = jax.scipy.signal.convolve(
                InfectedCases, reporting_delay, mode="full"
            )[:, : nDs]

            ExpectedCases = deterministic(
                "ExpectedCases", expected_cases.reshape((nRs, nDs))
            )

            # effectively handle missing values ourselves
            obs_probs = PsiCases / (
                PsiCases
                + ExpectedCases.reshape((nRs * nDs,))[
                    all_observed_active
                ]
            )
            ObservedCases = sample(
                "ObservedCases",
                NegativeBinomial(PsiCases, probs=obs_probs),
                obs=NewCases_data.reshape((nRs * nDs,))[
                    all_observed_active
                ],
            )

            if False:

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

        return run_model

