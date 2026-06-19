import numpy as np


def transform_to_fat_tailed(noise, power):
    """ Apply a power transformation (e.g., squaring the normal samples)"""
    fat_tailed_samples = np.sign(noise) * np.abs(noise)**power
    return fat_tailed_samples


def demand_sampler(nodal_demand, n_samples: int, random_seed: int = 42):
    """ Generate new demand samples
    for each bus based on the existing `nodal_demand`,
     by sampling around each existing demand value with added noise. """
    np.random.seed(random_seed)  # Reset the random seed for reproducibility
    nodal_demand_samples = []
    for s_idx in range(n_samples):  # For the current bus, sample around each existing load value
        # For each scenario, perturb the value by adding Gaussian noise
        # 5% noise as an example and transform it to a fat-tailed distribution
        noise = np.random.normal(loc=0, scale=0.05 * nodal_demand)
        nodal_demand_samples.append(nodal_demand  + transform_to_fat_tailed(noise, power=1.3))
    return nodal_demand_samples

