"""
epidemiological parameters
"""

import numpy as np
from scipy.stats import norm
import pprint
from tqdm import tqdm


def bootstrapped_negbinom_values(delays, n_bootstrap=250, n_rvs=int(1e7), truncation=64, filter_disp_outliers=True):
    """
    Fit negative binomial to n_bootstrapped sets of n_rv samples, each set of samples drawn randomly from the priors
    placed on the distributions in the delay array. e.g., this function is used to fit a single negative binomial
    distribution (with uncertainty) to the sum of the incubation period and onset to death delay.

    :param delays: list of distributions (with uncertainty), used to produce estimates.
    :param n_bootstrap: number of bootstrapped to perform
    :param n_rvs: number of samples to draw from each draw from prior
    :param truncation: maximum value to truncate to.
    :return: dictionary with uncertain NB values
    """
    means = np.zeros(n_bootstrap)
    disps = np.zeros(n_bootstrap)

    for seed in tqdm(range(n_bootstrap)):
        ep = EpidemiologicalParameters(seed)
        samples = np.zeros(n_rvs)

        for dist in delays:
            samples += ep.generate_dist_samples(dist, n_rvs, with_noise=True)

        samples[samples > truncation] = truncation
        means[seed] = np.mean(samples)
        disps[seed] = ((np.var(samples) - np.mean(samples)) / (np.mean(samples) ** 2)) ** -1

    if filter_disp_outliers:
        # especially for the fatality delay, this can be an issue.
        med_disp = np.median(disps)
        abs_deviations = np.abs(disps - med_disp)
        disps = disps[abs_deviations < 2 * np.median(abs_deviations)]

    ret = {
        'mean_mean': np.mean(means),
        'mean_sd': np.std(means),
        'disp_mean': np.mean(disps),
        'disp_sd': np.std(disps),
        'dist': 'negbinom'
    }

    return ret, means, disps


def ci_to_mean_sd(mean, ci, percent=0.95):
    sf = np.abs(norm.ppf((1 - percent) * 0.5))
    mean_sd = np.max(np.abs(ci - mean)) / sf
    return mean, mean_sd


class EpidemiologicalParameters():
    """
    Epidemiological Parameters Class

    Wrapper Class, contains information about the epidemiological parameters used in this project.
    """

    def __init__(self, seed=0, generation_interval=None, incubation_period=None, infection_to_fatality_delay=None,
                 infection_to_reporting_delay=None):
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

        :param numpy seed used for randomisation
        :param generation_interval: dictionary containing relevant distribution information
        :param incubation_period : dictionary containing relevant distribution information
        :param infection_to_fatality_delay: dictionaries containing relevant distribution information
        :param infection_to_reporting_delay: dictionaries containing relevant distribution information
        """
        if generation_interval is not None:
            self.generation_interval = generation_interval
        else:
            self.generation_interval = {
                'mean_mean': 5.06,
                'mean_sd': 0.3265,
                'sd_mean': 1.72,
                'sd_sd': 1.13,
                'source': 'mean: https://www.medrxiv.org/content/medrxiv/early/2020/06/19/2020.06.17.20133587.full.pdf'
                          'CoV: https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.17.2000257',
                'dist': 'gamma',
                'notes': 'mean_sd chosen to "fill CIs" from the medrxiv meta-analysis. sd_sd chosen for the same average'
                         'CoV from Ganyani et al, using the sd for the mean.'
            }

        if infection_to_fatality_delay is not None:
            self.infection_to_fatality_delay = infection_to_fatality_delay
        else:
            self.infection_to_fatality_delay = {
                'mean_mean': 20.88499992704522,
                'mean_sd': 1.6078311893096533,
                'disp_mean': 11.063081266841987,
                'disp_sd': 3.870808096034033,
                'source': 'incubation: Lauer et al, doi.org/10.7326/M20-0504, '
                          'onset-death hi',
                'dist': 'negbinom',
                'notes': 'Fitted as a bootstrapped NB.'
            }

        if incubation_period is not None:
            self.incubation_period = incubation_period
        else:
            self.incubation_period = {
                'mean_mean': 1.621,
                'mean_sd': 0.0684,
                'sd_mean': 0.418,
                'sd_sd': 0.0759,
                'source': 'sd: Lauer et al, doi.org/10.7326/M20-0504',
                'dist': 'lognorm',
                'notes': 'mean_mean, mean_sd chosen to "fill CIs" from the medrxiv meta-analysis.  '
                         '(log) sd, sd_sd taken from Lauer et al'
            }

        if infection_to_reporting_delay is not None:
            self.infection_to_reporting_delay = infection_to_reporting_delay
        else:
            self.infection_to_reporting_delay = {
                'mean_mean': 11.1,
                'mean_sd': 0.5,
                'disp_mean': 5.46,
                'disp_sd': 0.55,
                'source': 'incubation: Lauer et al, doi.org/10.7326/M20-0504'
                          'onset-reporting: Cereda et al, https://arxiv.org/abs/2003.09320',
                'dist': 'negbinom',
                'notes': 'Fitted as a bootstrapped NB.'
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
        if dist['dist'] == 'gamma':
            sd = np.random.normal(loc=dist['sd_mean'], scale=dist['sd_sd'] * with_noise)
            k = mean ** 2 / sd ** 2
            theta = sd ** 2 / mean
            samples = np.random.gamma(k, theta, size=nRVs)
        if dist['dist'] == 'gamma_cov':
            cov = np.random.normal(loc=dist['cov_mean'], scale=dist['cov_sd'] * with_noise)
            sd = cov * mean
            k = mean ** 2 / sd ** 2
            theta = sd ** 2 / mean
            samples = np.random.gamma(k, theta, size=nRVs)
        elif dist['dist'] == 'lognorm':
            sd = np.random.normal(loc=dist['sd_mean'], scale=dist['sd_sd'] * with_noise)
            # lognorm rv generated by e^z where z is normal
            samples = np.exp(np.random.normal(loc=mean, scale=sd, size=nRVs))
        elif dist['dist'] == 'negbinom':
            disp = np.random.normal(loc=dist['disp_mean'], scale=dist['disp_sd'] * with_noise)
            p = disp / (disp + mean)
            samples = np.random.negative_binomial(disp, p, size=nRVs)

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
        Generate reporting and fatality discretised delays using Monte Carlo integration.

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
        print('Infection to Reporting Delay')
        pprint.pprint(self.infection_to_reporting_delay)
        print('Infection to Fatality Delay')
        pprint.pprint(self.infection_to_fatality_delay)
        print('----------------------------------\n')

    def get_model_build_dict(self):
        """
        Grab parameters which can be conveniently passed to our model files.

        :return: param dict
        """
        ret = {
            'gi_mean_mean': self.generation_interval['mean_mean'],
            'gi_mean_sd': self.generation_interval['mean_sd'],
            'gi_sd_mean': self.generation_interval['sd_mean'],
            'gi_sd_sd': self.generation_interval['sd_sd'],
            'deaths_delay_mean_mean': self.infection_to_fatality_delay['mean_mean'],
            'deaths_delay_mean_sd': self.infection_to_fatality_delay['mean_sd'],
            'deaths_delay_disp_mean': self.infection_to_fatality_delay['disp_mean'],
            'deaths_delay_disp_sd': self.infection_to_fatality_delay['disp_sd'],
            'cases_delay_mean_mean': self.infection_to_reporting_delay['mean_mean'],
            'cases_delay_mean_sd': self.infection_to_reporting_delay['mean_sd'],
            'cases_delay_disp_mean': self.infection_to_reporting_delay['disp_mean'],
            'cases_delay_disp_sd': self.infection_to_reporting_delay['disp_sd'],
        }
        return ret
