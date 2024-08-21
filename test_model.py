import numpy as np
from scipy import stats
import pandas as pd

# Functie voor Likelihood Ratio Test (LRT)
def likelihood_ratio_test(log_likelihood_model1, log_likelihood_model2, df_model1, df_model2):
    """
    Voert een Likelihood Ratio Test (LRT) uit tussen twee modellen.
    
    Parameters:
    - log_likelihood_model1: Log-likelihood van model 1
    - log_likelihood_model2: Log-likelihood van model 2
    - df_model1: Vrijheidsgraden van model 1 (aantal parameters)
    - df_model2: Vrijheidsgraden van model 2 (aantal parameters)

    Returns:
    - p-waarde van de LRT
    """
    lr_stat = 2 * (log_likelihood_model2 - log_likelihood_model1)
    df_diff = df_model2 - df_model1
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value

# Functie voor A/B-test
def ab_test(r2_model1, r2_model2):
    """
    Voert een A/B-test uit om het verschil in prestaties tussen twee modellen te beoordelen op basis van hun R².

    Parameters:
    - r2_model1: R² van model 1
    - r2_model2: R² van model 2

    Returns:
    - p-waarde van de A/B-test
    """
    diff = np.mean(r2_model1) - np.mean(r2_model2)
    pooled_var = (np.var(r2_model1) + np.var(r2_model2)) / 2
    t_stat = diff / np.sqrt(pooled_var / len(r2_model1))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(r2_model1) - 1))
    return t_stat, p_value

# Laad de resultaten van de modellen in
# Voor dit voorbeeld, veronderstel dat de resultaten van de modellen al zijn opgeslagen in Excel-bestanden
model1_results = pd.read_excel('results_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx')
model2_results = pd.read_excel('results_PLScombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx')

# Log-likelihoods en aantal parameters van de modellen
log_likelihood_model1 = model1_results['Log_Likelihood'].values
log_likelihood_model2 = model2_results['Log_Likelihood'].values

# Aantal parameters (vrijheidsgraden) van de modellen
df_model1 = len(model1_results.columns) - 1  # -1 voor de niet-parameters
df_model2 = len(model2_results.columns) - 1

# R² van de modellen
r2_model1 = model1_results['R2_OutSample'].values
r2_model2 = model2_results['R2_OutSample'].values

# Voer de Likelihood Ratio Test uit
lr_stat, lr_p_value = likelihood_ratio_test(log_likelihood_model1.mean(), log_likelihood_model2.mean(), df_model1, df_model2)
print(f"Likelihood Ratio Test: LR Stat = {lr_stat}, p-value = {lr_p_value}")

# Voer de A/B-test uit
t_stat, ab_p_value = ab_test(r2_model1, r2_model2)
print(f"A/B Test: t-statistic = {t_stat}, p-value = {ab_p_value}")
