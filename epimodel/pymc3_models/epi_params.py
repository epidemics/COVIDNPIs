"""
epidemiological parameters
"""

import numpy as np
import pprint


class EpidemiologicalParameters():
    """
    Epidemiological Parameters Class

    Wrapper Class, contains information about the epidemiological parameters used in this project.
    """

    def __init__(self, seed=0, generation_interval=None, incubation_period=None, onset_reporting_delay=None,
                 onset_fatality_delay=None):
        """
        Constructor

        Input dictionarys corresponding to the relevant delay with the following fields:
            - mean_mean: mean of the mean value
            - mean_sd: sd of the mean value
            - sd_mean: mean of the sd value
            - sd_sd: sd of the sd value
            - source: str describing source information
            - distribution type: only 'gamma' and 'lognorm' are currently supported
            - notes: any other notes

        :param generation_interval: dictionary containing relevant distribution information
        :param incubation_period: dictionary containing relevant distribution information
        :param onset_reporting_delay: dictionary containing relevant distribution information
        :param onset_fatality_delay: dictionary containing relevant distribution information
        """
        if generation_interval is not None:
            self.generation_interval = generation_interval
        else:
            self.generation_interval = {
                'mean_mean': 3.635,
                'mean_sd': 0.7109,
                'sd_mean': 3.07532,
                'sd_sd': 0.769517,
                'source': 'https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.17.2000257',
                'dist': 'gamma',
                'notes': 'Exact Numbers taken from https://github.com/epiforecasts/EpiNow2'
            }

            self.generation_interval = {
                'mean_mean': 5.06,
                'mean_sd': 0.32,
                'sd_mean': 1.804,
                'sd_sd': 1e-5,
                'source': 'https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.17.2000257',
                'dist': 'gamma',
                'notes': 'Exact Numbers taken from https://github.com/epiforecasts/EpiNow2'
            }

        if incubation_period is not None:
            self.incubation_period = incubation_period
        else:
            self.incubation_period = {
                'mean_mean': 1.621,
                'mean_sd': 0.064,
                'sd_mean': 0.518,
                'sd_sd': 0.0691,
                'source': 'Lauer et al, doi.org/10.7326/M20-0504',
                'dist': 'lognorm',
                'notes': 'Exact Numbers taken from https://github.com/epiforecasts/EpiNow2'
            }

            # self.incubation_period = {
            #     'mean_mean': 1.624,
            #     'mean_sd': 0.064,
            #     'sd_mean': 0.518,
            #     'sd_sd': 0.0691,
            #     'source': 'Lauer et al, doi.org/10.7326/M20-0504',
            #     'dist': 'lognorm',
            #     'notes': 'Exact Numbers taken from https://github.com/epiforecasts/EpiNow2'
            # }

        if onset_reporting_delay is not None:
            self.onset_reporting_delay = onset_reporting_delay
        else:
            self.onset_reporting_delay = {
                'mean_mean': 0.974,
                'mean_sd': 0.1583,
                'sd_mean': 1.4662,
                'sd_sd': 0.120,
                'source': 'https://github.com/beoutbreakprepared/nCoV2019',
                'dist': 'lognorm',
                'notes': 'Produced using linelist data and the EpiNow2 Repo, fitting a lognorm variable.'
                         '200 Bootstraps with 250 samples each.'
            }

            self.onset_reporting_delay = {
                'mean_mean': 5.2,
                'mean_sd': 0.6025,
                'sd_mean': 4.78,
                'sd_sd': 1e-5,
                'source': 'https://github.com/beoutbreakprepared/nCoV2019',
                'dist': 'gamma',
                'notes': 'Produced using linelist data and the EpiNow2 Repo, fitting a lognorm variable.'
                         '200 Bootstraps with 250 samples each.'
            }

        if onset_fatality_delay is not None:
            self.onset_fatality_delay = onset_fatality_delay
        else:
            self.onset_fatality_delay = {
                'mean_mean': 2.28,
                'mean_sd': 0.0685,
                'sd_mean': 0.763,
                'sd_sd': 0.0537,
                'source': 'https://github.com/epiforecasts/covid-rt-estimates',
                'dist': 'lognorm',
                'notes': 'taken from data/onset_to_death_delay.rds'
            }

            self.onset_fatality_delay = {
                'mean_mean': 16.71,
                'mean_sd': 0.7,
                'sd_mean': 7.52,
                'sd_sd': 1e-5,
                'source': 'https://github.com/epiforecasts/covid-rt-estimates',
                'dist': 'gamma',
                'notes': 'taken from data/onset_to_death_delay.rds'
            }

        self.seed = seed

    def generate_dist_samples(self, dist, nRVs, with_noise):
        """
        Generate samples from given distribution.

        :param dist: Distribution dictionary to use.
        :param nRVs: number of random variables to sample
        :param with_noise: if true, add noise to distributions, else do not.
        :return: samples
        """
        # specify seed because everything here is random!!
        np.random.seed(self.seed)
        mean = np.random.normal(loc=dist['mean_mean'], scale=dist['mean_sd'] * with_noise)
        sd = np.random.normal(loc=dist['sd_mean'], scale=dist['sd_sd'] * with_noise)
        if dist['dist'] == 'gamma':
            k = mean ** 2 / sd ** 2
            theta = sd ** 2 / mean
            samples = np.random.gamma(k, theta, size=nRVs)
        elif dist['dist'] == 'lognorm':
            # lognorm rv generated by e^z where z is normal
            samples = np.exp(np.random.normal(loc=mean, scale=sd, size=nRVs))

        return samples

    def discretise_samples(self, samples, max_int):
        """
        Discretise a set of samples to form a pmf, truncating to max.

        :param samples: Samples to discretize.
        :param max: Truncation.
        :return: pmf - discretised distribution.
        """
        bins = np.arange(-1.0, float(max_int))
        bins[2:] += 0.5

        counts = np.histogram(samples, bins)[0]
        # normalise
        pmf = counts / np.sum(counts)
        pmf = pmf.reshape((1, pmf.size))

        return pmf

    def generate_pmf_statistics_str(self, delay_prob):
        """
        Make mean and variance of delay string.

        :param delay_prob: delay to compute statistics of.
        :return: Information string.
        """
        n_max = delay_prob.size
        mean = np.sum([(i) * delay_prob[0, i] for i in range(n_max)])
        var = np.sum([(i ** 2) * delay_prob[0, i] for i in range(n_max)]) - mean ** 2
        return f'mean: {mean:.3f}, sd: {var ** 0.5:.3f}, max: {n_max}'

    def generate_gi(self, with_noise=True):
        """
        Generate generation interval parameters.

        Note: this uses the random seed associated with the EpidemiologicalParameters() object, and will be consistent.

        :param: with_noise: boolean - if True, add noise to estimates
        :return: Mean, sd of generation interval.
        """
        np.random.seed(self.seed)
        dist = self.generation_interval
        mean = np.random.normal(loc=dist['mean_mean'], scale=dist['mean_sd'] * with_noise)
        sd = np.random.normal(loc=dist['sd_mean'], scale=dist['sd_sd'] * with_noise)
        print(f'Generation Interval: mean: {mean :.3f}, sd: {sd :.3f}')
        return mean, sd

    def generate_reporting_and_fatality_delays(self, nRVs=int(1e7), with_noise=True, max_reporting=32, max_fatality=48):
        """
        Generate reporting and fatality delays using Monte Carlo integration.

        Note: this uses the random seed associated with the EpidemiologicalParameters() object, and will be consistent.

        :param nRVs: int - number of random variables used for integration
        :param max_reporting: int - reporting delay truncation
        :param with_noise: boolean. If true, use noisy values for the incubation period etc, otherwise use the means.
        :param max_fatality: int - death delay trunction
        :return: reporting_delay, fatality_delay tuple
        """
        incubation_period_samples = self.generate_dist_samples(self.incubation_period, nRVs, with_noise)
        reporting_samples = self.generate_dist_samples(self.onset_reporting_delay, nRVs, with_noise)
        fatality_samples = self.generate_dist_samples(self.onset_fatality_delay, nRVs, with_noise)

        print(f'Raw: reporting delay mean {np.mean(incubation_period_samples + reporting_samples)}')
        print(f'Raw: fatality delay mean {np.mean(incubation_period_samples + fatality_samples)}')
        reporting_delay = self.discretise_samples(incubation_period_samples + reporting_samples, max_reporting)
        fatality_delay = self.discretise_samples(incubation_period_samples + fatality_samples, max_fatality)
        print(f'Generated Reporting Delay: {self.generate_pmf_statistics_str(reporting_delay)}')
        print(f'Generated Fatality Delay: {self.generate_pmf_statistics_str(fatality_delay)}')

        return reporting_delay, fatality_delay

    def summarise_parameters(self):
        """
        Print summary of parameters.
        """
        print('Epidemiological Parameters Summary\n'
              '----------------------------------\n')
        print('Generation Interval')
        pprint.pprint(self.generation_interval)
        print('Incubation Period')
        pprint.pprint(self.incubation_period)
        print('Symptom Onset to Reporting Delay')
        pprint.pprint(self.onset_reporting_delay)
        print('Symptom Onset to Fatality Delay')
        pprint.pprint(self.onset_fatality_delay)
        print('----------------------------------\n')
