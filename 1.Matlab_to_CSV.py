# Import packages needed
import os
import pandas as pd
import scipy.io as sio
import numpy as np
import re
from sklearn.impute import KNNImputer

# Set directories
in_directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_multilead/"
out_directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_multilead_clean_data/"


def matlab_to_csv(directory, output_directory):
    """"Transform the matlab file into CSV file"""
    # Create output directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Transform matlab file into a csv file
    for filename in os.listdir(directory):
        # Load data
        filepath = os.path.join(directory, filename)
        data = sio.loadmat(filepath)
        var_names = list(data.keys())
        biomarkers = data[var_names[3]]

        # Transform numpy array into dataframe
        biomarkers_df = pd.DataFrame(biomarkers)

        # Name the columns with the appropiate names
        column_names = ['Pavg', 'AreaQRSabs', 'qt', 'QTc', 'ST_0', 'ST_60', 'ST_slope', 'pq', 'AreaTiniTpeak',
                        'AreaTpeakTfin', 'STCsimetria', 'Tsimetria', 'QRSd', 'RMS40', 'LAS', '15', '16', 'Beat',
                        'Lead']

        biomarkers_df.columns = column_names

        # Generate the CSV files and save them
        file = filename.split("_")[1][9:]
        type = filename.split("_")[2][:9]
        output_csv_filename = file + "_" + type + ".csv"
        output_csv_filepath = os.path.join(output_directory, output_csv_filename)
        biomarkers_df.to_csv(output_csv_filepath, decimal=',', index=False)


def process_and_impute_data(filepath):
    """Clean the data of infinities NAN and impute the missing values"""
    # Compute percentages of NAN values per column grouping by lead
    data_frame = pd.read_csv(filepath, decimal=',')
    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame = data_frame[data_frame.iloc[:, -1] != 0]
    nan_percentage = data_frame.isna().groupby(data_frame[data_frame.columns[-1]]).mean() * 100
    delete_columns = []

    # Delete the columns that has >30% of missing values and impute the rest with KNNImputer using default parameters
    # n_neighbors=5, metric='euclidean', weights='uniform'
    for column, percent in nan_percentage.items():
        if percent.values[0] > 30:
            delete_columns.append(column)

    if delete_columns:
        data_frame.drop(columns=delete_columns, inplace=True)

    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(data_frame)
    data_frame = pd.DataFrame(imputed_data, columns=data_frame.columns)

    # Add the ID of the patients
    file_number = int(re.search(r'BH(\d+)_', filepath).group(1))
    data_frame.insert(0, 'BH', file_number)

    # Add the condition of the patients, 1 for Symptomatic and 0 for Asymptomatic
    if file_number in [3, 23, 29, 33, 44, 46, 98, 99, 101, 111, 122, 123, 126, 131, 132, 135, 143, 148, 156, 158, 169,
                       172, 183, 185]:
        data_frame.insert(1, 'Symptomatic', 1)
    else:
        data_frame.insert(1, 'Symptomatic', 0)

    # Correct some wrong identifications of the leads
    if file_number < 88:
        data_frame[data_frame.columns[-1]] = data_frame[data_frame.columns[-1]].replace(
            {8: 10, 9: 8, 10: 11, 11: 9})

    data_frame.to_csv(filepath, decimal=',', index=False)


matlab_to_csv(in_directory, out_directory)

for filename in os.listdir(out_directory):
    csv_filepath = os.path.join(out_directory, filename)
    process_and_impute_data(csv_filepath)

print("Imputation Done")
