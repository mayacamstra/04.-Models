import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standardize_df(df_data:pd.DataFrame) -> pd.DataFrame:
    '''
        Standardize values in the dataframe by subtracting means
        and dividing by the standard deviations column-wise.
        Uses the StandardScaler class from sklearn.
        
        Returns
        --
            pd.DataFrame with standardized data values.
    '''

    # Standardize the feature matrix
    scaler = StandardScaler()
    std_values = scaler.fit_transform(df_data)
#    std_values = (df_data - np.mean(df_data, axis=0))/np.std(df_data,axis=0)
    return pd.DataFrame(std_values, index=df_data.index, columns=df_data.columns)


def apply_pca(std_data:pd.DataFrame, num_factors:int) -> pd.DataFrame:
    '''
        Factor extraction from standardized data using PCA.

        Params
        --
            std_data: pd.DataFrame containing detrended
            and standardized time series data of
            multiple economic variables.
            num_factors: int that specifies number of 
            factors to return, should be smaller than 
            the cross-sectional dimension of the data.

        Returns
        -- 
            self.df_factors: pd.DataFrame consisting of factors.
                Number of columns: num_factors.
                Number of rows: same as df input.
    '''
    pca = PCA(n_components = num_factors)
    principal_components = pca.fit_transform(std_data)

    df_factors = pd.DataFrame(index=std_data.index, data=principal_components, 
                        columns=[f'Factor {i+1}' for i in range(num_factors)])
    return df_factors



# Import the CSV containing key countries and variables of the OFS as a Pandas DataFrame.
path = r"C:\Users\raviark\Repositories\Temp_FCS\FCS 2024 - Data PCA.csv"

data = pd.read_csv(path, index_col =0)
data = data.transpose()

# Standardize data and apply PCA
std_data = standardize_df(data)
factors = apply_pca(std_data, num_factors=9)
factors = factors.iloc[1:]
std_factors = standardize_df(factors)
std_factors.to_csv(r"C:\Users\raviark\Repositories\Temp_FCS\Factors_output.csv")



