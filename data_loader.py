import pandas as pd
import numpy as np
  
def load_data(file_path):
    """
    Load data from an Excel file and convert the columns to monthly periods.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: The loaded data with monthly period columns.
    """
    df_data = pd.read_excel(file_path, engine='openpyxl', index_col=0)
    # Convert columns to Period with monthly frequency
    df_data.columns = pd.to_datetime(df_data.columns, format='%d/%m/%Y').to_period('M')
    return df_data

def filter_data(df_data):
    """
    Apply the Christiano-Fitzgerald filter to the data.

    Parameters:
    df_data (pd.DataFrame): The original data.

    Returns:
    pd.DataFrame: The filtered data.
    """
    matrix = df_data.to_numpy()
    result = Christiano_Fitzgerald_filter(matrix, [2, 192], adjust_level=False, generate_external_sides=[10, 10])
    return pd.DataFrame(index=df_data.index, columns=df_data.columns, data=result["component 0"])

def load_combined_data(static_file_path, forward_file_path):
    """
    Load and combine static and forward data from Excel files, converting columns to monthly periods.

    Parameters:
    static_file_path (str): The path to the static data Excel file.
    forward_file_path (str): The path to the forward data Excel file.

    Returns:
    pd.DataFrame: The combined data with monthly period columns.
    """
    df_static = load_data(static_file_path)
    df_forward = load_data(forward_file_path)
    return pd.concat([df_static, df_forward])

# Christiano-Fitzgerald filter functions...

def Christiano_Fitzgerald_filter(data, barriers, adjust_level=False, generate_external_sides=[0, 0]):
    """
    Apply the Christiano-Fitzgerald filter to the data.
    
    Parameters:
    data (np.ndarray): The input data matrix.
    barriers (list): The frequency barriers for the filter.
    adjust_level (bool): Whether to adjust the level after filtering.
    generate_external_sides (list): Number of external sides to generate.

    Returns:
    dict: Filtered components and other information.
    """
    time_series_matrix = data.copy()

    if max(generate_external_sides) > 0:
        first_column = time_series_matrix[:, :1]
        last_column = time_series_matrix[:, -1:]
        time_series_matrix = np.concatenate(
            [np.tile(first_column, [1, generate_external_sides[0]]), time_series_matrix,
             np.tile(last_column, [1, generate_external_sides[1]])], axis=1)

    variables, dates = time_series_matrix.shape
    original_dates = data.shape[1]
    times = np.arange(1, dates + 1)

    frequencies = [2 * np.pi / barrier if barrier != 0 else 0 for barrier in barriers]
    frequencies.sort()

    len_t = len(times)

    filter_dict = {}
    filter_dict["frequencies"] = frequencies

    for index, w_1 in enumerate(frequencies[:-1]):
        w_2 = frequencies[index + 1]
        g_0 = (w_2 - w_1) / np.pi
        g_l = [(np.sin(w_2 * l) - np.sin(w_1 * l)) / (np.pi * l) for l in np.arange(1, len_t)]
        g_l = [g_0] + g_l
        g_last = [(-0.5 * g_0 - np.sum(g_l[-i:])) for i in range(1, len_t + 1)]
        provisional_matrix = generate_wanted_matrix(g_l)
        provisional_matrix[-1, :] = g_last
        g_first = np.sum(provisional_matrix[1:, :], axis=0)
        provisional_matrix[0, :] = g_first
        y = time_series_matrix @ provisional_matrix
        if max(generate_external_sides) > 0:
            y = y[:, generate_external_sides[0]:-generate_external_sides[1]]

        if adjust_level:
            if index == 0:
                total_level = np.zeros(y.shape)

            total_level += y

            if w_1 == frequencies[-2]:
                diff = time_series_matrix[:, generate_external_sides[0] + 5] - total_level[:, 5]
                diff = np.tile(diff.reshape([variables, 1]), [1, original_dates])
                total_level += diff
                filter_dict["component 0"] += diff

        filter_dict["component " + str(index)] = y
        if adjust_level: 
            filter_dict["Total"] = total_level

        filter_dict["provisional_matrix " + str(index)] = provisional_matrix
    return filter_dict

def generate_wanted_matrix(values):
    size = len(values)
    reverse_values = values[1:]
    reverse_values.reverse()
    new_values = reverse_values + values
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        row = new_values[size - 1 - i:2 * size - i - 1]
        matrix[i, :] = np.array(row)
    return matrix