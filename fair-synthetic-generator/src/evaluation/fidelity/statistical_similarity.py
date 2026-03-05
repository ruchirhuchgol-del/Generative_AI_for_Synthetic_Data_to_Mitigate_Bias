import numpy as np
from scipy.stats import ks_2samp

def calculate_statistical_similarity(real_data, synthetic_data):
    """Calculates the Kolmogorov-Smirnov test for each column."""
    similarities = {}
    for i in range(real_data.shape[1]):
        statistic, p_value = ks_2samp(real_data[:, i], synthetic_data[:, i])
        similarities[f"col_{i}"] = {
            "ks_statistic": statistic,
            "p_value": p_value
        }
    return similarities

def calculate_mse(real_data, synthetic_data):
    """Calculates Mean Squared Error between real and synthetic data distributions."""
    return np.mean((np.mean(real_data, axis=0) - np.mean(synthetic_data, axis=0))**2)
