import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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

# Define paths
path_static = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Static_Data_v1(Smaller).csv"
path_combined = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Combined_Full_Data_v3.csv"

# Transpose Data Frames
data_static = pd.read_csv(path_static, index_col =0)
data_static = data_static.transpose()

data_combined = pd.read_csv(path_combined, index_col = 0, sep =';')
data_combined = data_combined.transpose()

# Apply standardization and PCA
std_data_static = standardize_df(data_static)
factors_static = apply_pca(std_data_static, num_factors=9)
std_factors_static = standardize_df(factors_static)

std_data_combined = standardize_df(data_combined)
factors_combined = apply_pca(std_data_combined, num_factors=9)
std_factors_combined = standardize_df(factors_combined)

# Save factor Data Frames
std_factors_static.to_csv(r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\PCA\PCA_factors_static.csv")
std_factors_combined.to_csv(r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\PCA\PCA_factors_combined.csv")

# Plotting Factor 1,2,3,4 from both datasets
plt.figure(figsize=(10, 5))
plt.plot(factors_static.index, factors_static['Factor 4'], label='Static', marker='o', linestyle='-')
plt.plot(factors_combined.index, factors_combined['Factor 4'], label='Combined', marker='x', linestyle='--')

plt.title('Comparison of Factor 4 Over Time')
plt.xlabel('Date')
plt.ylabel('Factor 4 Values')
plt.legend()
plt.grid(True)
plt.show()

# One big plot with distributions and boxplots
# Setting up the figure and axes
fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(15, 40))  # Adjust the figsize based on your screen and preferences

# Loop over each factor to create both a histogram and a box plot
for i, factor in enumerate(std_factors_static.columns, start=1):
    # Histograms
    ax = axes[i-1, 0]  # Row i-1, Column 0
    ax.hist(std_factors_static[factor], bins=30, alpha=0.7, label='Static', color='blue')
    ax.hist(std_factors_combined[factor], bins=30, alpha=0.7, label='Combined', color='orange')
    ax.set_title(f'Histogram of {factor}')
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Box plots
    ax = axes[i-1, 1]  # Row i-1, Column 1
    ax.boxplot([std_factors_static[factor], std_factors_combined[factor]], labels=['Static', 'Combined'])
    ax.set_title(f'Box Plot of {factor}')
    ax.set_ylabel('Values')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


