# Import packages needed
import os
import pandas as pd

# set directories
in_directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_multilead_clean_data/"
out_directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_Transposed/"


def combine_csv_files(input_directory):
    """Concatenate all CSV files into a single DataFrame"""
    # Create output directory
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    files = []

    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        files.append(pd.read_csv(filepath, decimal=','))

    # Remove columns that have missing values.
    # We just work with the Biomarkers that appear in all the files.
    combined_df = pd.concat(files, ignore_index=True)
    data_frame = combined_df.dropna(axis=1)
    return data_frame


def transpose_data_frame(data_frame, output_directory):
    """Transpose the data and relate each biomarker to its lead and beat """
    data_frame['Lead'] = data_frame['Lead'].astype(int)
    data_frame['Beat'] = data_frame['Beat'].astype(int)

    # Keep the condition of Symptomatic for every patient
    symptomatic_values = data_frame.groupby('BH').nth(0)['Symptomatic']
    transposed_data = []

    # Transpose the data
    for bh in data_frame['BH'].unique():
        patient_data = data_frame[data_frame['BH'] == bh]
        new_columns = {}

        for col in patient_data.columns[2:-2]:
            for lead in patient_data['Lead'].unique():
                for beat in patient_data['Beat'].unique():
                    mask = (patient_data['Lead'] == lead) & (patient_data['Beat'] == beat)
                    value = patient_data.loc[mask, col].iloc[0]
                    new_column_name = f'{col}_lead{lead}_beat{beat}'
                    new_columns[new_column_name] = value

        new_columns['BH'] = bh
        transposed_data.append(new_columns)

    data_frame_transposed = pd.DataFrame(transposed_data)
    data_frame_transposed['Symptomatic'] = symptomatic_values.values

    # Add number of patients and symptoms in the firsts columns
    columns = ['BH'] + ['Symptomatic'] + [col for col in data_frame_transposed if col not in ['BH', 'Symptomatic']]
    data_frame_transposed = data_frame_transposed[columns]

    output_filepath = os.path.join(output_directory, "Data_Transposed.csv")
    data_frame_transposed.to_csv(output_filepath, index=False)
    print("Data frame has been transposed.")


def filter_and_impute_data(input_directory, nan_percentage):
    """Filter columns by NaN values and impute missing values"""
    data_frame = pd.read_csv(input_directory)

    # Calculate the maximum number of NaN values allowed
    max_nan_count = int(len(data_frame) * nan_percentage)

    # Remove columns with more NaN values than allowed
    data_frame_filtered = data_frame.dropna(axis=1, thresh=max_nan_count)

    # Impute missing values with median grouped by 'Symptomatic'
    data_frame_imputed = data_frame_filtered.copy()
    for col in data_frame_filtered.columns[2:]:
        median_by_symptomatic = data_frame_filtered.groupby('Symptomatic')[col].median()
        for symptomatic_value in median_by_symptomatic.index:
            mask = (data_frame_imputed['Symptomatic'] == symptomatic_value) & data_frame_imputed[col].isna()
            data_frame_imputed.loc[mask, col] = median_by_symptomatic[symptomatic_value]

    output_filepath = os.path.join(out_directory, "Todos_filtered_imputed.csv")
    data_frame_imputed.to_csv(output_filepath, index=False)
    print("Data frame has been filtered and imputed.")


def process_csv_data(input_directory, output_directory):
    """Call the functions to transform the CSV files"""
    combined_df = combine_csv_files(input_directory)
    transpose_data_frame(combined_df, output_directory)
    input_filepath = os.path.join(output_directory, "Data_Transposed.csv")
    filter_and_impute_data(input_filepath, max_nan_percentage)


max_nan_percentage = 0.3
process_csv_data(in_directory, out_directory)
