# Import packages needed
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
import statsmodels.api as sm

# Set directory
directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_multilead_clean_data/"


def combine_csv_files(input_directory, output_filepath):
    """Concatenate all the files into a single one"""
    dfs = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_directory, filename)
            dfs.append(pd.read_csv(filepath, decimal=','))

    combined_df = pd.concat(dfs, ignore_index=True)
    data_frame = combined_df.dropna(axis=1)
    data_frame.to_csv(output_filepath, decimal=',', index=False)
    print("Concatenation done")


def detect_outliers_and_normality(data_frame, alpha=0.05):
    """Detect outliers, plot boxplots, QQ plots, and histograms, and perform KS test."""
    for column in data_frame.columns[2:-2]:
        Q1 = data_frame[column].quantile(0.25)
        Q3 = data_frame[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the bounds for outliers
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR

        # Identify the outliers
        outliers = biomarkers_df[(biomarkers_df[column] < lower_bound) | (biomarkers_df[column] > upper_bound)]

        # Box plot for every column to see the outliers
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data_frame[column], color="lightblue")
        plt.title(f"Box Plot for {column}")
        plt.axvline(x=lower_bound, color='r', linestyle='--', label=f"Lower Bound ({lower_bound:.2f})")
        plt.axvline(x=upper_bound, color='g', linestyle='--', label=f"Upper Bound ({upper_bound:.2f})")
        plt.legend()
        plt.xlabel('Value')
        plt.show()

        # QQ plot to see if the values follow a normal distribution
        sm.qqplot(data_frame[column], line='45')
        plt.title(f"QQ Plot for {column}")
        plt.show()

        # Kolmogorov-Smirnov (KS) test to verify if the data follows a normal distribution
        ks_statistic, ks_p_value = kstest(data_frame[column], 'norm')
        print(f"KS statistic for {column}: {ks_statistic}")
        print(f"P-value for {column}: {ks_p_value}")

        if ks_p_value > alpha:
            print("Accept null hypothesis. Values follow a normal distribution.")
        else:
            print("Reject null hypothesis. Values don't follow a normal distribution.")

        # Histogram to see how the values are distributed
        sns.histplot(data_frame[column], kde=True, color='blue', bins=20)
        plt.title(f'Histogram of {column} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def calculate_spearman_correlation(data_frame):
    """Calculate and plot Spearman correlation matrix."""
    columns_for_spearman = data_frame.columns[2:-2]
    corr_matrix_spearman = data_frame[columns_for_spearman].corr(method='spearman')

    print(f"Spearman correlation matrix:")
    print(corr_matrix_spearman)

    # Heatmap for the Spearman correlation matrix
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    sns.heatmap(
        corr_matrix_spearman,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        ax=ax
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
    )

    ax.tick_params(labelsize=10)
    plt.title('Spearman Correlation Heatmap')
    plt.show()


# Set the output file path
output_filepath = "Todos.csv"

combine_csv_files(directory, output_filepath)
biomarkers_df = pd.read_csv(output_filepath, decimal=',')

# Descriptive statistics
relevant_columns = biomarkers_df.iloc[:, 2:-2]
statistics = relevant_columns.describe()
print("Descriptive statistics:")
print(statistics)

detect_outliers_and_normality(biomarkers_df)
calculate_spearman_correlation(biomarkers_df)
