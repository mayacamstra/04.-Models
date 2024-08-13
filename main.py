import pandas as pd
from data_loader import load_data, load_combined_data, filter_data
from utils import RMSE
from individual_model import IndividualModel
from train_model import train_and_evaluate_model

def main(data_type='static', method='PCA', num_factors=9):
    """
    Main function to run the model on either the 'static' or 'combined' dataset.

    Parameters:
    data_type (str): Choose between 'static' or 'combined'.
    method (str): Factor extraction method, either 'PCA' or 'PLS'.
    num_factors (int): Number of factors to extract.
    """
    
    if data_type == 'static':
        file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
        df_data = load_data(file_path)
    elif data_type == 'combined':
        static_file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
        forward_file_path = 'C:/Thesis/03. Data/Final version data/Forward.xlsx'
        df_data = load_combined_data(static_file_path, forward_file_path)
    else:
        raise ValueError("data_type must be 'static' or 'combined'")
    
    filtered_df = filter_data(df_data)
    
    # Model initialiseren
    model = IndividualModel(filtered_df, num_factors=num_factors, method=method)

    # Data splitsen voor training en validatie
    DATE_VALIDATE = pd.Period('2020-01', freq='M')
    date_index = filtered_df.columns.get_loc(DATE_VALIDATE)

    Y_train_PCA = filtered_df.iloc[:, :date_index]
    REGRESSION_STEP = 12
    Y_train_other = Y_train_PCA.iloc[:, REGRESSION_STEP:]
    Y_reg_train = filtered_df.iloc[:, :date_index + 1 - REGRESSION_STEP]

    # Model trainen en evalueren
    B_matrix, C_matrix, r2_insample, beta_const, rmse_value = train_and_evaluate_model(
        model, Y_train_other, Y_reg_train, Y_train_PCA.iloc[:, date_index:], Y_reg_train)

    # RMSE resultaten voorbereiden voor export
    if data_type == 'static':
        variables_to_display = df_data.index[:66]
        rmse_df = pd.DataFrame({'Variable': variables_to_display, 'RMSE': rmse_value[:66]})
    else:  # combined
        variables_to_display = df_data.index[:66]
        rmse_df = pd.DataFrame({'Variable': variables_to_display, 'RMSE': rmse_value[:66]})

    # Resultaten weergeven
    print(rmse_df)
    print(f"R2 in-sample: {r2_insample}")
    print(f"ElasticNet intercept: {beta_const}")

    # Opslaan van RMSE-resultaten naar Excel
    output_filename = f'rmse_{data_type}_{method}.xlsx'
    rmse_df.to_excel(output_filename, index=False)
    print(f"RMSE resultaten opgeslagen in {output_filename}")

if __name__ == "__main__":
    main(data_type='static', method='PCA')  # Pas deze parameters aan om het script te draaien voor verschillende scenario's
    