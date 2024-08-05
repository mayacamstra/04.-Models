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

# Import the CSV containing only static variables as a Pandas DataFrame.
path = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Static_Data_v1(Smaller).csv"

data = pd.read_csv(path, index_col =0)
data = data.transpose()

std_data = standardize_df(data)
factors = apply_pca(std_data, num_factors=9)
std_factors = standardize_df(factors)
std_factors.to_csv(r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\PCA\PCA_factors_static.csv")

# Import the CSV containing both static and foward-looking variables as a Pandas DataFrame.
path5 = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Combined_Full_Data_v3.csv"

data_combined = pd.read_csv(path5, index_col = 0, sep =';')
data_combined = data_combined.transpose()

std_data_combined = standardize_df(data_combined)
factors_combined = apply_pca(std_data_combined, num_factors=9)
std_factors_combined = standardize_df(factors_combined)
std_factors_combined.to_csv(r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\PCA\PCA_factors_combined.csv")

# Start comparative analysis of the obtained factors; are they statistically significantly different from each other?
# Observe the different factor dataframes:
print(std_factors_combined.head())  # --> factors are different from observing heads
print(std_factors.head())
for factor in std_factors.columns:
    print(f"{factor} variance in static: {np.var(std_factors[factor])}")
    print(f"{factor} variance in combined: {np.var(std_factors_combined[factor])}")

assert std_factors.shape == std_factors_combined.shape, "The DataFrames have different shapes!"

# Visualizations
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(std_factors['Factor 2'], bins=30, alpha=0.7, label='Static')
plt.hist(std_factors_combined['Factor 2'], bins=30, alpha=0.7, label='Combined')
plt.title('Comparison of Factor 2 Distributions')
plt.xlabel('Factor 2 Values')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot([std_factors['Factor 2'], std_factors_combined['Factor 2']], labels=['Static', 'Combined'])
plt.title('Box Plot of Factor 2')
plt.ylabel('Factor 2 Values')
plt.show()

# Setting up the figure and axes
fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(15, 40))  # Adjust the figsize based on your screen and preferences

# Loop over each factor to create both a histogram and a box plot
for i, factor in enumerate(std_factors.columns, start=1):
    # Histograms
    ax = axes[i-1, 0]  # Row i-1, Column 0
    ax.hist(std_factors[factor], bins=30, alpha=0.7, label='Static', color='blue')
    ax.hist(std_factors_combined[factor], bins=30, alpha=0.7, label='Combined', color='orange')
    ax.set_title(f'Histogram of {factor}')
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Box plots
    ax = axes[i-1, 1]  # Row i-1, Column 1
    ax.boxplot([std_factors[factor], std_factors_combined[factor]], labels=['Static', 'Combined'])
    ax.set_title(f'Box Plot of {factor}')
    ax.set_ylabel('Values')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Perform a paired t-test
p_values = {}
for factor in std_factors.columns:
    stat, p_value = ttest_rel(std_factors[factor], std_factors_combined[factor])
    p_values[factor] = p_value

# Print p-values for each factor
for factor, p in p_values.items():
    print(f"Factor {factor}: p-value = {p:.4f}")

# Wilcoxon Signed-Rank Test
from scipy.stats import wilcoxon

p_values_wilcoxon = {}
for factor in std_factors.columns:
    stat, p_value = wilcoxon(std_factors[factor], std_factors_combined[factor])
    p_values_wilcoxon[factor] = p_value

# Print p-values for each factor
for factor, p in p_values_wilcoxon.items():
    print(f"Factor {factor} (Wilcoxon): p-value = {p:.4f}")

# Adjust for Multiple Comparisons ---> CAN DELETE
from statsmodels.stats.multitest import multipletests

# Example using Bonferroni correction ---> CAN DELETE
p_adjusted = multipletests(list(p_values_wilcoxon.values()), alpha=0.05, method='bonferroni')

# Print adjusted p-values  ---> CAN DELETE
for i, factor in enumerate(std_factors.columns):
    print(f"Factor {factor} (Adjusted): p-value = {p_adjusted[1][i]:.4f}")

# Visualization of p-values  --> CAN DELETE
import matplotlib.pyplot as plt
import numpy as np

# Plotting -log10 of p-values  --> CAN DELETE
p_vals = list(p_values.values())
plt.bar(range(len(p_vals)), -np.log10(p_vals), tick_label=list(p_values.keys()))
plt.ylabel('-Log10 p-value')
plt.title('Significance of Differences between Factors')
plt.axhline(y=-np.log10(0.05), color='r', linestyle='dashed')  # Threshold for p=0.05
plt.show()

# Overview of how the factors look
print(std_factors.head())
print(std_factors.tail())
print(std_factors_combined.head())
print(std_factors_combined.tail())

# Need to adjust the date formats, by means of indexes
