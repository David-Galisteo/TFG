# Import packages needed
import pandas as pd

# Set directory
excel_file = "/home/david/bioinformatica/TFG/Anonimizado_bueno_1.xlsx"
encoded_csv_file = "/home/david/bioinformatica/TFG/Anonimizado_bueno_1_encoded.csv"
csv_file1 = "/home/david/bioinformatica/Pruebas/Biomarcadores_Transposed/Todos_filtered_imputed.csv"
csv_file2 = "/home/david/bioinformatica/TFG/Anonimizado_bueno_1_encoded.csv"
merged_csv_file = "archivo_fusionado.csv"


def process_excel_to_csv (excel_file, output_csv):
    """Performs the one hot encoding to incorporate clinical features"""
    df = pd.read_excel(excel_file)

    one_hot_encoded = pd.get_dummies(df['Ethnic'])
    one_hot_encoded_gender = pd.get_dummies(df['Gender'])
    one_hot_encoded_symptoms = pd.get_dummies(df['Symptoms_bef_Dg_wo_vagal_wo_other (cardiogenic)'], prefix='Symptoms')

    df = df.drop(['Ethnic', 'Gender', 'Symptoms_bef_Dg_wo_vagal_wo_other (cardiogenic)'], axis=1)

    # Convert True to 1 and False to 0
    one_hot_encoded = one_hot_encoded.astype(int)
    one_hot_encoded_gender = one_hot_encoded_gender.astype(int)
    one_hot_encoded_symptoms = one_hot_encoded_symptoms.astype(int)

    # Concatenate the encoded columns to the original DataFrame
    df = pd.concat([df, one_hot_encoded, one_hot_encoded_gender, one_hot_encoded_symptoms], axis=1)

    df.to_csv(output_csv, index=False)

    print(f"CSV file saved: {output_csv}")


def merge_csv_files(csv_file1, csv_file2, output_csv):
    """Concatenate the both files into a single one"""
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Merge DataFrames based on BH
    merged_df = pd.merge(df1, df2, on="BH", how="inner")
    merged_df.to_csv(output_csv, index=False)

    print(f"Merged CSV file saved: {output_csv}")


process_excel_to_csv(excel_file, encoded_csv_file)

merge_csv_files(csv_file1, csv_file2, merged_csv_file)

