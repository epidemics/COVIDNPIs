"""
:code:`models.py`

Contains a variety of models of NPI effectiveness, all subclassed from BaseCMModel. 
"""
import numpy as np
import pymc3 as pm

import theano.tensor as T
import theano.tensor.signal.conv as C

from epimodel import EpidemiologicalParameters
from .base_model import BaseCMModel


class DefaultModel(BaseCMModel):
    """
    Default Model

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Initialiser.
        """
        super().__init__(data, cm_plot_style, name, model)

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32):
        """
        Build NPI effectiveness model

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
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
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # build R_0 prior
            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs))
            self.RegionR = pm.Deterministic("RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            # load CMs active, compute log-R reduction and region log-R based on NPIs active
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.LogRReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.LogRReduction

            # convert R into growth rates
            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            # exclude 40 days of noise, slight increase in runtime.
            if growth_noise_scale > 0:
                self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale,
                                                  shape=(self.nRs, self.nDs - 40))
                self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                                   shape=(self.nRs, self.nDs - 40))
            else:
                self.GrowthCasesNoise = T.zeros((self.nRs, self.nDs - 40))
                self.GrowthDeathsNoise = T.zeros((self.nRs, self.nDs - 40))

            self.GrowthCases = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthCasesNoise)
            self.GrowthDeaths = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthDeathsNoise)

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean,
                                         cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            # convolve with delay to produce expectations
            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            # output distribution
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            # Deaths
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            # convolve with delay to production reports
            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            # death output distribution
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DeathsOnlyModel(BaseCMModel):
    """
    Deaths only model.

    Identical to the default model, other than modelling only deaths.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    deaths_truncation=48, **kwargs):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_truncation: maximum death delay
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.GrowthReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.GrowthReduction
            )

            # convert R into growth rates
            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(
                self.ExpectedLogR))

            self.GrowthNoiseDeaths = pm.Normal('GrowthNoiseDeaths', 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs - 40))

            self.Growth = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthNoiseDeaths)

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic('Infected', pm.math.exp(self.InfectedDeaths_log))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)

            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.Infected,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5)

            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths])

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )


class CasesOnlyModel(BaseCMModel):
    """
    Cases only model.

    Identical to the default model, other than modelling only cases.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    cases_truncation=32, **kwargs):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction
            )

            # convert R into growth rates
            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(
                self.ExpectedLogR))

            self.GrowthNoiseCases = pm.Normal('GrowthNoiseCases', 0, growth_noise_scale,
                                              shape=(self.nRs, self.nDs - 40))

            self.Growth = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthNoiseCases)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.Growth.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean,
                                         cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_confirmed.reshape(
                (self.nRs, self.nDs)))

            self.PsiCases = pm.HalfNormal('PsiCases', 5)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )


class NoisyRModel(BaseCMModel):
    """
    Noisy-R Model.
    
    This is the same as the default model, but adds noise to R_t before converting this to the growth rate, g_t. In the 
    default model, noise is added to g_t.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, **kwargs):
        """
        Build NPI effectiveness model

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
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
        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs))
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.RegionLogR = pm.math.log(self.RegionR)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.GrowthReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogRCases = pm.Normal(
                'ExpectedLogRCases',
                T.reshape(self.RegionLogR, (self.nRs, 1)) - self.GrowthReduction,
                R_noise_scale,
                shape=(self.nRs, self.nDs)
            )

            self.ExpectedLogRDeaths = pm.Normal(
                'ExpectedLogRDeaths',
                T.reshape(self.RegionLogR, (self.nRs, 1)) - self.GrowthReduction,
                R_noise_scale,
                shape=(self.nRs, self.nDs)
            )

            # convert R into growth rates
            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.GrowthCases = gi_beta * (
                    pm.math.exp(self.ExpectedLogRCases / gi_alpha) - T.ones_like(self.ExpectedLogRCases))

            self.GrowthDeaths = gi_beta * (
                    pm.math.exp(self.ExpectedLogRDeaths / gi_alpha) - T.ones_like(self.ExpectedLogRDeaths))

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean,
                                         cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class AdditiveModel(BaseCMModel):
    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10,
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale. For this model, this is the concentration parameter
                                dirichlet distribution, same for all NPIs.
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
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
        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            self.AllBeta = pm.Dirichlet('AllBeta', cm_prior_scale * np.ones((self.nCMs + 1)), shape=(self.nCMs + 1,))
            self.CM_Beta = pm.Deterministic('CM_Beta', self.AllBeta[1:])
            self.Beta_hat = pm.Deterministic('Beta_hat', self.AllBeta[0])
            self.CMReduction = pm.Deterministic('CMReduction', self.CM_Beta)

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            active_cm_reduction = T.reshape(self.CM_Beta, (1, self.nCMs, 1)) * (
                    T.ones_like(self.ActiveCMs) - self.ActiveCMs)

            growth_reduction = T.sum(active_cm_reduction, axis=1) + self.Beta_hat

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.log(T.exp(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))) * growth_reduction)
            )

            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(self.ExpectedLogR))

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale,
                                              shape=(self.nRs, self.nDs - 40))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs - 40))

            self.GrowthCases = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthCasesNoise)
            self.GrowthDeaths = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthDeathsNoise)

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean,
                                         cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # learn the output noise for this.
            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DifferentEffectsModel(BaseCMModel):
    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    alpha_noise_scale=0.1, deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9,
                    deaths_delay_disp_sd=1, cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5,
                    cases_delay_disp_sd=1, deaths_truncation=48, cases_truncation=32, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale. For this model, this is the concentration parameter
                                dirichlet distribution, same for all NPIs.
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
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

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.AllCMAlpha = pm.Normal('AllCMAlpha',
                                        T.reshape(self.CM_Alpha, (1, self.nCMs)).repeat(self.nRs, axis=0),
                                        alpha_noise_scale,
                                        shape=(self.nRs, self.nCMs)
                                        )

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            active_cm_reduction = T.reshape(self.AllCMAlpha, (self.nRs, self.nCMs, 1)) * self.ActiveCMs
            growth_reduction = T.sum(active_cm_reduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction,
            )

            # convert R into growth rates
            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            # exclude 40 days of noise, slight increase in runtime.
            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale,
                                              shape=(self.nRs, self.nDs - 40))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs - 40))

            self.GrowthCases = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthCasesNoise)
            self.GrowthDeaths = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthDeathsNoise)

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.GrowthCases.cumsum(axis=1))
            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean,
                                         cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))
            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DiscreteRenewalModel(BaseCMModel):
    """
    Discrete Renewal Model.

    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a discrete renewal model, adding noise on R.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed', R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, gi_truncation=28, conv_padding=7,
                    gi_mean_mean=5, gi_mean_sd=0.3, gi_sd_mean=2, gi_sd_sd=0.3, **kwargs):
        """
        Build NPI effectiveness model

        :param gi_sd_sd: gi std prior std
        :param gi_sd_mean: gi std prior mean
        :param gi_mean_sd: gi mean prior std
        :param gi_mean_mean: gi mean prior mean
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param R_noise_scale: multiplicative noise scale, now placed on R!
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
        :param gi_truncation: truncation used for generation interval discretisation
        :param conv_padding: padding for renewal process
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        # ep = EpidemiologicalParameters()
        # gi_s = ep.generate_dist_samples(ep.generation_interval, nRVs=int(1e8), with_noise=False)
        # GI = ep.discretise_samples(gi_s, gi_truncation).flatten()
        # GI_rev = GI[::-1].reshape((1, 1, GI.size)).repeat(2, axis=0)

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            GI_dist = pm.Gamma.dist(alpha=gi_alpha, beta=gi_beta)

            bins = np.zeros(gi_truncation + 1)
            bins[1:] = np.arange(gi_truncation)
            bins[2:] += 0.5
            bins[:2] += 1e-5

            cdf_vals = T.exp(GI_dist.logcdf(bins))
            pmf = cdf_vals[1:] - cdf_vals[:-1]
            GI_rev = T.repeat(T.reshape(pmf[::-1] / T.sum(pmf), (1, 1, gi_truncation)), 2, axis=0)

            self.RReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.RReduction,
                                          (1, self.nRs, self.nDs)).repeat(2, axis=0)

            if R_noise_scale > 0:
                self.LogRNoise = pm.Normal('LogRNoise', 0, R_noise_scale, shape=(2, self.nRs, self.nDs - 40))
            else:
                self.LogRNoise = T.zeros((2, self.nRs, self.nDs - 40))

            self.LogR = pm.Deterministic('LogR', T.inc_subtensor(self.ExpectedLogR[:, :, 30:-10], self.LogRNoise))

            self.InitialSize_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(2, self.nRs))

            infected = T.zeros((2, self.nRs, self.nDs + gi_truncation))
            infected = T.set_subtensor(infected[:, :, (gi_truncation - conv_padding):gi_truncation],
                                       pm.math.exp(self.InitialSize_log.reshape((2, self.nRs, 1)).repeat(
                                           conv_padding, axis=2)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, :, d].reshape((2, self.nRs, 1)) * infected[:, :, d:(d + gi_truncation)] * GI_rev,
                    axis=2)
                infected = T.set_subtensor(infected[:, :, d + gi_truncation], val)

            res = infected

            self.InfectedCases = pm.Deterministic(
                'InfectedCases',
                res[0, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[1, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean,
                                         cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.NewCases = pm.Data('NewCases',
                                    self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
            )

            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )


class DeathsOnlyDiscreteRenewalModel(BaseCMModel):
    """
    Deaths Only Discrete Renewal Model.

    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a discrete renewal model, adding noise on R.

    It also models deaths also
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed', R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    deaths_truncation=48, gi_truncation=28, conv_padding=7,
                    gi_mean_mean=5, gi_mean_sd=0.3, gi_sd_mean=2, gi_sd_sd=0.3, **kwargs):
        """
        Build NPI effectiveness model

        :param gi_sd_sd: gi std prior std
        :param gi_sd_mean: gi std prior mean
        :param gi_mean_sd: gi mean prior std
        :param gi_mean_mean: gi mean prior mean
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param R_noise_scale: multiplicative noise scale, now placed on R!
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        :param gi_truncation: truncation used for generation interval discretisation
        :param conv_padding: padding for renewal process
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.build_gi_prior(gi_mean_mean, gi_mean_sd, gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            GI_dist = pm.Gamma.dist(alpha=gi_alpha, beta=gi_beta)

            bins = np.zeros(gi_truncation + 1)
            bins[1:] = np.arange(gi_truncation)
            bins[2:] += 0.5
            bins[:2] += 1e-5

            cdf_vals = T.exp(GI_dist.logcdf(bins))
            pmf = cdf_vals[1:] - cdf_vals[:-1]
            GI_rev = T.reshape(pmf[::-1] / T.sum(pmf), (1, gi_truncation))

            self.RReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.RReduction,
                                          (self.nRs, self.nDs))

            if R_noise_scale > 0:
                self.LogRNoise = pm.Normal('LogRNoise', 0, R_noise_scale, shape=(self.nRs, self.nDs - 40))
            else:
                self.LogRNoise = T.zeros((self.nRs, self.nDs - 40))

            self.LogR = pm.Deterministic('LogR', T.inc_subtensor(self.ExpectedLogR[:, 30:-10], self.LogRNoise))

            self.InitialSize_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))

            infected = T.zeros((self.nRs, self.nDs + gi_truncation))
            infected = T.set_subtensor(infected[:, (gi_truncation - conv_padding):gi_truncation],
                                       pm.math.exp(self.InitialSize_log.reshape((self.nRs, 1)).repeat(
                                           conv_padding, axis=1)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, d].reshape((self.nRs, 1)) * infected[:, d:(d + gi_truncation)] * GI_rev,
                    axis=-1).reshape((self.nRs,))
                infected = T.set_subtensor(infected[:, d + gi_truncation], val)

            res = infected

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[:, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                          deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )


class DiscreteRenewalFixedGIModel(BaseCMModel):
    """
    Discrete Renewal Model.
    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a discrete renewal model, adding noise on R.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed', R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, gi_truncation=28, conv_padding=7, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param R_noise_scale: multiplicative noise scale, now placed on R!
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
        :param gi_truncation: truncation used for generation interval discretisation
        :param conv_padding: padding for renewal process
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        # discretise once!
        ep = EpidemiologicalParameters()
        gi_s = ep.generate_dist_samples(ep.generation_interval, nRVs=int(1e8), with_noise=False)
        GI = ep.discretise_samples(gi_s, gi_truncation).flatten()
        GI_rev = GI[::-1].reshape((1, 1, GI.size)).repeat(2, axis=0)

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.RReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.RReduction,
                                          (1, self.nRs, self.nDs)).repeat(2, axis=0)

            self.LogRNoise = pm.Normal('LogRNoise', 0, R_noise_scale, shape=(2, self.nRs, self.nDs-40))
            self.LogR = T.inc_subtensor(self.ExpectedLogR[:, :, 30:-10], self.LogRNoise)

            self.InitialSize_log = pm.TruncatedNormal('InitialSizeCases_log', 0, 50, shape=(2, self.nRs), upper=10)

            infected = T.zeros((2, self.nRs, self.nDs + gi_truncation))
            infected = T.set_subtensor(infected[:, :, (gi_truncation - conv_padding):gi_truncation],
                                       pm.math.exp(self.InitialSize_log.reshape((2, self.nRs, 1)).repeat(
                                           conv_padding, axis=2)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, :, d].reshape((2, self.nRs, 1)) * infected[:, :, d:(d + gi_truncation)] * GI_rev,
                    axis=2)
                infected = T.set_subtensor(infected[:, :, d + gi_truncation], val)

            res = infected

            self.InfectedCases = pm.Deterministic(
                'InfectedCases',
                res[0, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[1, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.build_cases_delay_prior(cases_delay_mean_mean, cases_delay_mean_sd, cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = pm.NegativeBinomial.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            self.build_deaths_delay_prior(deaths_delay_mean_mean, deaths_delay_mean_sd, deaths_delay_disp_mean,
                                         deaths_delay_disp_sd)
            deaths_delay_dist = pm.NegativeBinomial.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.NewCases = pm.Data('NewCases',
                                    self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
            )

            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )

class DiscreteRenewalLegacyModel(BaseCMModel):
    """
    Discrete Renewal Model.
    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a discrete renewal model, adding noise on R.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed', R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, gi_truncation=28, conv_padding=7, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param R_noise_scale: multiplicative noise scale, now placed on R!
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
        :param gi_truncation: truncation used for generation interval discretisation
        :param conv_padding: padding for renewal process
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        # discretise once!
        ep = EpidemiologicalParameters()
        gi_s = ep.generate_dist_samples(ep.generation_interval, nRVs=int(1e8), with_noise=False)
        GI = ep.discretise_samples(gi_s, gi_truncation).flatten()
        GI = np.array(
            [0, 0.04656309, 0.08698277, 0.1121656, 0.11937737, 0.11456359,
             0.10308026, 0.08852893, 0.07356104, 0.059462, 0.04719909,
             0.03683025, 0.02846977, 0.02163222, 0.01640488, 0.01221928,
             0.00903811, 0.00670216, 0.00490314, 0.00361434, 0.00261552,
             0.00187336, 0.00137485, 0.00100352, 0.00071164, 0.00050852,
             0.00036433, 0.00025036]
        )
        GI_rev = GI[::-1].reshape((1, 1, GI.size)).repeat(2, axis=0)

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.RReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.RReduction,
                                          (1, self.nRs, self.nDs)).repeat(2, axis=0)

            self.LogRNoise = pm.Normal('LogRNoise', 0, R_noise_scale, shape=(2, self.nRs, self.nDs-40))
            self.LogR = T.inc_subtensor(self.ExpectedLogR[:, :, 30:-10], self.LogRNoise)

            self.InitialSize_log = pm.TruncatedNormal('InitialSizeCases_log', 0, 50, shape=(2, self.nRs), upper=10)

            infected = T.zeros((2, self.nRs, self.nDs + gi_truncation))
            infected = T.set_subtensor(infected[:, :, (gi_truncation - conv_padding):gi_truncation],
                                       pm.math.exp(self.InitialSize_log.reshape((2, self.nRs, 1)).repeat(
                                           conv_padding, axis=2)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, :, d].reshape((2, self.nRs, 1)) * infected[:, :, d:(d + gi_truncation)] * GI_rev,
                    axis=2)
                infected = T.set_subtensor(infected[:, :, d + gi_truncation], val)

            res = infected

            self.InfectedCases = pm.Deterministic(
                'InfectedCases',
                res[0, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[1, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            fatality_delay = np.array([0.00000000e+00, 1.64635735e-06, 3.15032703e-05, 1.86360977e-04,
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
            fatality_delay = fatality_delay.reshape((1, fatality_delay.size))

            reporting_delay = np.array([0., 0.0252817, 0.03717965, 0.05181224, 0.06274125,
                                            0.06961334, 0.07277174, 0.07292397, 0.07077184, 0.06694868,
                                            0.06209945, 0.05659917, 0.0508999, 0.0452042, 0.03976573,
                                            0.03470891, 0.0299895, 0.02577721, 0.02199923, 0.01871723,
                                            0.01577148, 0.01326564, 0.01110783, 0.00928827, 0.0077231,
                                            0.00641162, 0.00530572, 0.00437895, 0.00358801, 0.00295791,
                                            0.0024217, 0.00197484])

            reporting_delay = reporting_delay.reshape((1, reporting_delay.size))


            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.NewCases = pm.Data('NewCases',
                                    self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
            )

            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )