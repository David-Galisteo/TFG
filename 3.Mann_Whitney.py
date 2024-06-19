# Import packages needed
import pandas as pd
from scipy.stats import mannwhitneyu

# Set directory
directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_multilead_clean_data/Todos.csv"

# Define constant
alpha = 0.05


def mann_whitney_test(directory):
    """Perform the Mann-Whitney U test for each biomarker"""
    data = pd.read_csv(directory, decimal=',')

    biomarkers = ['Pavg', 'AreaQRSabs', 'qt', 'ST_0', 'ST_60', 'ST_slope', 'QRSd', 'RMS40']

    # Split data into symptomatic and asymptomatic groups
    symptomatic = data[data['Symptomatic'] == 1]
    asymptomatic = data[data['Symptomatic'] == 0]

    # Perform Mann-Whitney U test
    results = {}
    for biomarker in biomarkers:
        stat, p_value = mannwhitneyu(symptomatic[biomarker], asymptomatic[biomarker])
        results[biomarker] = {'U-Statistic': stat, 'p-value': p_value}
        if p_value < alpha:
            print(f'{biomarker} shows significant differences (p-value = {p_value:.5f}).')
        else:
            print(f'{biomarker} does not show significant differences (p-value = {p_value:.5f}).')

    results_df = pd.DataFrame(results).T

    return results_df


results_df = mann_whitney_test(directory)

print(results_df)
