import numpy as np


def transform_to_fat_tailed(noise, power=1.6):
    """ Apply a power transformation (e.g., squaring the normal samples)"""
    fat_tailed_samples = np.sign(noise) * np.abs(noise)**power
    return fat_tailed_samples


def simple_demand_sampler(mean_demand_value,
                          n_samples: int,
                          noise_level : float = 0.05,
                          fat_tailed : bool= False,
                          random_seed: int = 42):
    """ Generate new demand samples  for each bus based on the existing `nodal_demand`,
     by sampling around each existing demand value with added noise. """
    np.random.seed(random_seed)  # Reset the random seed for reproducibility
    nodal_demand_samples = []
    for s_idx in range(n_samples):  # For the current bus, sample around each existing load value
        # For each scenario, add Gaussian noise + fat-tailed component if
        if fat_tailed:
            noise = np.random.normal(loc=0, scale=noise_level * mean_demand_value)
            nodal_demand_samples.append(mean_demand_value + transform_to_fat_tailed(noise))
        else:
            noise = np.random.normal(loc=0, scale=noise_level * mean_demand_value)
            nodal_demand_samples.append(mean_demand_value + noise)
    return nodal_demand_samples

