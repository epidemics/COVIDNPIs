"""
Models file

Contains several different model implementations.
"""
import copy

import numpy as np
import pymc3 as pm

import theano.tensor as T
import theano.tensor.signal.conv as C

from .base_model import BaseCMModel
from .epi_params import EpidemiologicalParameters

SI_ALPHA = 1
SI_BETA = 1


class DefaultModel(BaseCMModel):
    """
    Default Model

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor function.

        At the moment, just calls the BaseCMModel Constructor
        """
        super(DefaultModel, self).__init__(data, cm_plot_style, name, model)

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.2,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]])):
        """
        Build PyMC3 model.

        :param R_hyperprior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. defaults to 0.2
        :param fatality_delay: infection to fatality array delay.
        :param reporting_delay: infection to reporting array delay.
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
            gi_beta = generation_interval_mean / generation_interval_sigma ** 2
            gi_alpha = generation_interval_mean ** 2 / generation_interval_sigma ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs))

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth + self.GrowthCasesNoise)
            self.GrowthDeaths = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth + self.GrowthDeathsNoise)

            self.Dispersion = pm.HalfNormal("Dispersion", 0.1)

            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

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
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            # Deaths
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

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
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DefaultModelFixedDispersion(BaseCMModel):
    """
    Default Model

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor function.

        At the moment, just calls the BaseCMModel Constructor
        """
        super(DefaultModelFixedDispersion, self).__init__(data, cm_plot_style, name, model)

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.2,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]])):
        """
        Build PyMC3 model.

        :param R_hyperprior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. defaults to 0.2
        :param fatality_delay: infection to fatality array delay.
        :param reporting_delay: infection to reporting array delay.
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
            gi_beta = generation_interval_mean / generation_interval_sigma ** 2
            gi_alpha = generation_interval_mean ** 2 / generation_interval_sigma ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs))

            self.GrowthCases = self.ExpectedGrowth + self.GrowthCasesNoise
            self.GrowthDeaths = self.ExpectedGrowth + self.GrowthDeathsNoise

            self.Dispersion = 0.0025

            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

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
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            # Deaths
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

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
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DefaultModelLognorm(BaseCMModel):
    """
    Default Model

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor function.

        At the moment, just calls the BaseCMModel Constructor
        """
        super(DefaultModelLognorm, self).__init__(data, cm_plot_style, name, model)

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.2,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]])):
        """
        Build PyMC3 model.

        :param R_hyperprior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. defaults to 0.2
        :param fatality_delay: infection to fatality array delay.
        :param reporting_delay: infection to reporting array delay.
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
            gi_beta = generation_interval_mean / generation_interval_sigma ** 2
            gi_alpha = generation_interval_mean ** 2 / generation_interval_sigma ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs))

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth + self.GrowthCasesNoise)
            self.GrowthDeaths = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth + self.GrowthDeathsNoise)

            self.OutputNoiseScale = pm.HalfNormal('OutputNoiseScale', 0.5)

            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

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
            self.LogObservedCases = pm.Normal(
                "LogObservedCases",
                pm.math.log(self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active]),
                self.OutputNoiseScale,
                shape=(len(self.all_observed_active),),
                observed=pm.math.log(self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active])
            )

            # Deaths
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

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
            self.LogObservedDeaths = pm.Normal(
                "LogObservedDeaths",
                pm.math.log(self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]),
                self.OutputNoiseScale,
                shape=(len(self.all_observed_deaths),),
                observed=pm.math.log(self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths])
            )


class DeathsOnlyModel(BaseCMModel):
    """
    Deaths only model.

    Identical to the default model, other than modelling only deaths.
    """

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.2,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]])):
        """
        Build PyMC3 model.

        :param R_prior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. defaults to 0.2
        :param fatality_delay: infection to fatality array delay.
        :param reporting_delay: infection to reporting array delay. Note, this isn't actually used here, but kept to
                                keep the same function signature as other model classes.
        """
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
            gi_beta = generation_interval_mean / generation_interval_sigma ** 2
            gi_alpha = generation_interval_mean ** 2 / generation_interval_sigma ** 2

            self.ExpectedGrowth = gi_beta * (np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(
                self.ExpectedLogR))

            self.Growth = pm.Normal('Growth',
                                    self.ExpectedGrowth,
                                    growth_noise_scale,
                                    shape=(self.nRs, self.nDs))

            self.InitialSize_log = pm.Normal('InitialSize_log', -6, 100, shape=(self.nRs,))
            self.Infected_log = pm.Deterministic('Infected_log', T.reshape(self.InitialSize_log, (
                self.nRs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic('Infected', pm.math.exp(self.Infected_log))

            expected_deaths = C.conv2d(
                self.Infected,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nORs, self.nDs)))

            self.Dispersion = pm.HalfNormal('Dispersion', 0.1)

            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths])

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.Phi,
                shape=(len(self.observed_days),),
                observed=self.NewDeaths
            )


class CasesOnlyModel(BaseCMModel):
    """
    Cases only model.

    Identical to the default model, other than modelling only cases.
    """

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.2,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]])):
        """
        Build PyMC3 model.

        :param R_prior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. defaults to 0.2
        :param fatality_delay: infection to fatality array delay. Note, this isn't actually used here, but kept to
                                keep the same function signature as other model classes.
        :param reporting_delay: infection to reporting array delay.
        """
        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.Det(
                'GrowthReduction', T.sum(self.ActiveCMReduction, axis=1), plot_trace=False
            )

            self.ExpectedLogR = self.Det(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nORs, 1)) - self.GrowthReduction,
                plot_trace=False,
            )

            # convert R into growth rates
            gi_beta = generation_interval_mean / generation_interval_sigma ** 2
            gi_alpha = generation_interval_mean ** 2 / generation_interval_sigma ** 2
            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(self.ExpectedLogR))

            self.Growth = pm.Normal('Growth',
                                    self.ExpectedGrowth,
                                    growth_noise_scale,
                                    shape=(self.nRs, self.nDs))

            self.InitialSize_log = pm.Normal('InitialSize_log', -6, 100, shape=(self.nRs,))
            self.Infected_log = pm.Deterministic('Infected_log', T.reshape(self.InitialSize_log, (
                self.nRs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic('Infected', pm.math.exp(self.Infected_log))

            expected_confirmed = C.conv2d(
                self.Infected,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_confirmed.reshape(
                (self.nRs, self.nDs)))

            self.Dispersion = pm.HalfNormal('Phi', 0.1)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )


class DiscreteRenewalModel(BaseCMModel):
    """
    Discrete Renewal Model.

    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a renewal model, with noise on R.
    """

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.7,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]]), conv_padding=7):
        """
        Build PyMC3 model.

        :param R_prior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. defaults to 0.7. Note, for this model, this noise
                                    scale is actually applied to R!
        :param fatality_delay: infection to fatality array delay.
        :param reporting_delay: infection to reporting array delay.
        :param conv_padding: padding used for GI discretisation padding.
        """

        # somewhat hacky, but this model actually needs the discretised generation interval delay
        ep = EpidemiologicalParameters(generation_interval={
            'mean_mean': generation_interval_mean,
            'mean_sd': generation_interval_sigma,
            'sd_mean': 1.0,
            'sd_sd': 1.0,
        })
        s = ep.generate_dist_samples(ep.generation_interval, False)
        GI = ep.discretise_samples(s, 28)
        GI_rev = GI[::-1].reshape((1, 1, GI.size)).repeat(2, axis=0)

        # TODO: need to check this actually works, I replaced nOD with nD, might break something here.
        # also, i might have just made things slower. Not sure
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

            self.LogR = pm.Normal('LogR', self.ExpectedLogR, growth_noise_scale, shape=(2, self.nORs, self.nODs))

            self.InitialSize_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(2, self.nORs))

            filter_size = GI.size
            conv_padding = 7

            infected = T.zeros((2, self.nORs, self.nODs + self.SI.size))
            infected = T.set_subtensor(infected[:, :, (filter_size - conv_padding):filter_size],
                                       pm.math.exp(self.InitialSize_log.reshape((2, self.nORs, 1)).repeat(
                                           conv_padding, axis=2)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, :, d].reshape((2, self.nRs, 1)) * infected[:, :, d:(d + GI.size)] * GI_rev,
                    axis=2)
                infected = T.set_subtensor(infected[:, :, d + GI.size], val)

            res = infected

            self.InfectedDeaths = pm.Deterministic(
                'InfectedCases',
                res[0, :, GI.size:].reshape((self.nRs, self.nDs))
            )

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[1, :, GI.size:].reshape((self.nRs, self.nDs))
            )

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            self.Dispersion = pm.HalfNormal('Dispersion', 0.1)

            self.NewCases = pm.Data('NewCases',
                                    self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedDeaths = pm.NegativeBinomial(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )

            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
            )


class NoisyRModel(BaseCMModel):
    """
    Noisy-R Model.
    
    This is the same as the default model, but adds noise to R_t before converting this to the growth rate, g_t. In the 
    default model, noise is added to g_t.
    """

    def build_model(self, R_prior_mean=3.25, cm_prior_scale=10, cm_prior='skewed',
                    generation_interval_mean=5, generation_interval_sigma=2, growth_noise_scale=0.7,
                    fatality_delay=np.array([[1.0]]), reporting_delay=np.array([[1.0]])):
        """
        Build PyMC3 model.
 
        :param R_prior_mean: mean for R hyperprior.
        :param cm_prior_scale: prior scale parameter. See BaseCMModel.build_npi_prior()
        :param cm_prior: prior type
        :param generation_interval_mean: assumed fixed mean for gamma generation interval
        :param generation_interval_sigma: assumed fixed sd for gamma generation interval
        :param growth_noise_scale: growth noise scale hyperparamter. Here, this scale is actually applied to R.
        :param fatality_delay: infection to fatality array delay.
        :param reporting_delay: infection to reporting array delay.
        """
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
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                growth_noise_scale,
                shape=(self.nRs, self.nDs)
            )

            self.ExpectedLogRDeaths = pm.Normal(
                'ExpectedLogRDeaths',
                T.reshape(self.RegionLogR, (self.nORs, 1)) - self.GrowthReduction,
                growth_noise_scale,
                shape=(self.nRs, self.nDs)
            )

            # convert R into growth rates
            gi_beta = generation_interval_mean / generation_interval_sigma ** 2
            gi_alpha = generation_interval_mean ** 2 / generation_interval_sigma ** 2

            self.GrowthCases = gi_beta * (
                    pm.math.exp(self.ExpectedLogRCases / gi_alpha) - T.ones_like(self.ExpectedLogRCases))

            self.GrowthDeaths = gi_beta * (
                    pm.math.exp(self.ExpectedLogRDeaths / gi_alpha) - T.ones_like(self.ExpectedLogRDeaths))

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nORs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # learn the output noise for this.
            self.Dispersion = pm.HalfNormal('Dispersion', 0.1)

            # effectively handle missing values ourselves
            self.ObservedCases = pm.NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

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
                alpha=1 / self.Dispersion,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class CMCombined_Final_NoNoise(BaseCMModel):
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
                                         4.02211774e-04, 3.35345193e-04, 2.82450401e-04, 2.38109993e-04]
                                        )
        self.DelayProbDeaths = self.DelayProbDeaths.reshape((1, self.DelayProbDeaths.size))

        self.CMDelayCut = 30

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

            self.GrowthCases = pm.Deterministic("GrowthCases", self.ExpectedGrowth)
            self.GrowthCases = pm.Deterministic("GrowthDeaths", self.ExpectedGrowth)

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


class CMCombined_Final_DifDelays(BaseCMModel):
    def __init__(
            self, data, cm_plot_style=None, name="", model=None
    ):
        super().__init__(data, cm_plot_style, name=name, model=model)

        # infection --> confirmed delay
        self.DelayProbCasesShort = np.array([0., 0.04086903, 0.05623389, 0.07404812, 0.08464692,
                                             0.08861931, 0.08750149, 0.08273123, 0.07575679, 0.06766597,
                                             0.05910415, 0.05093048, 0.04321916, 0.03622008, 0.03000523,
                                             0.02472037, 0.02016809, 0.01637281, 0.01318903, 0.01057912,
                                             0.00844349, 0.0067064, 0.00529629, 0.00416558, 0.00327265,
                                             0.00255511, 0.00200011, 0.00155583, 0.00120648, 0.00093964,
                                             0.00072111, 0.00055606])
        self.DelayProbCasesLong = np.array([0., 0.01690821, 0.02602795, 0.03772294, 0.0474657,
                                            0.05484009, 0.05969648, 0.06231737, 0.06292536, 0.0619761,
                                            0.05983904, 0.05677383, 0.05311211, 0.04914501, 0.04502909,
                                            0.04085248, 0.03682251, 0.03290895, 0.02924259, 0.02585378,
                                            0.02274018, 0.01993739, 0.01739687, 0.01511531, 0.01309569,
                                            0.01130081, 0.00972391, 0.00832998, 0.00716289, 0.00610338,
                                            0.00520349, 0.00443053])

        self.DelayProbCases = np.stack([self.DelayProbCasesShort, self.DelayProbCasesLong]).reshape(
            (2, 1, self.DelayProbCasesShort.size))

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

        testing_indx = self.d.CMs.index("Symptomatic Testing")
        self.short_rs = np.nonzero(np.sum(data.ActiveCMs[:, testing_indx, :], axis=-1) > 1)[0]
        self.long_rs = np.nonzero(np.sum(data.ActiveCMs[:, testing_indx, :], axis=-1) < 1)[0]
        data.ActiveCMs[:, testing_indx, :] = 0

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

            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nORs,))
            self.InfectedCases_log = pm.Deterministic("InfectedCases_log", T.reshape(self.InitialSizeCases_log, (
                self.nORs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(self.InfectedCases_log))

            expected_cases = C.conv2d(
                self.InfectedCases,
                self.DelayProbCases,
                border_mode="full"
            )[:, :, :self.nDs]

            # automatically calculates which are short and which are long, and grabs from the correct convolution.
            # probably not the most efficient implementation
            expected_cases_temp = T.zeros_like(self.InfectedCases)
            expected_cases_temp = T.set_subtensor(expected_cases_temp[self.short_rs, :],
                                                  expected_cases[0, self.short_rs, :].reshape(
                                                      (len(self.short_rs), self.nDs)))
            expected_cases_temp = T.set_subtensor(expected_cases_temp[self.long_rs, :],
                                                  expected_cases[1, self.long_rs, :].reshape(
                                                      (len(self.long_rs), self.nDs)))

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases_temp)

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

    def build_model(self, R_hyperprior_mean=3.25, cm_prior_conc=1,
                    serial_interval_mean=SI_ALPHA / SI_BETA
                    ):
        with self.model:
            self.AllBeta = pm.Dirichlet("AllBeta", cm_prior_conc * np.ones((self.nCMs + 1)), shape=(self.nCMs + 1,))
            self.CM_Beta = pm.Deterministic("CM_Beta", self.AllBeta[1:])
            self.Beta_hat = pm.Deterministic("Beta_hat", self.AllBeta[0])
            self.CMReduction = pm.Deterministic("CMReduction", self.CM_Beta)

            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic("RegionR", R_hyperprior_mean + self.RegionLogR_noise * self.HyperRVar)

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
                T.log(T.exp(T.reshape(pm.math.log(self.RegionR), (self.nORs, 1))) * self.GrowthReduction),
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

    def plot_effect(self, save_fig=True, output_dir="./out", x_min=-100, x_max=100):

        # local imports, since this plotting only happens here
        import seaborn as sns
        from matplotlib.font_manager import FontProperties
        import matplotlib.pyplot as plt
        fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")
        sns.set_style("ticks")

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
        self.RegionVariationNoise = 0.1

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

            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nORs), )
            self.RegionR = pm.Deterministic("RegionR", R_hyperprior_mean + self.RegionLogR_noise * self.HyperRVar)

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
                T.reshape(pm.math.log(self.RegionR), (self.nORs, 1)) - self.GrowthReduction,
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
