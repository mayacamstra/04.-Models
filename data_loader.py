import pandas as pd

def load_data(file_path):
    """
    Load data from an Excel file and convert the columns to monthly periods.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: The loaded data with datetime columns.
    """
    df_data = pd.read_excel(file_path, engine='openpyxl', index_col=0)
    df_data.columns = pd.to_datetime(df_data.columns, format='%d/%m/%Y').to_period('M')
    return df_data
