import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load dat
df = pd.read_csv('table_8_non_index_results.csv')

# Prepare vars
# X = Volume (Billions), Y = EFQ
df['Vol_Bn'] = df['Total Executed Shares'] / 1e9
X = df['Vol_Bn']
y = df['Mean EFQ']

# Add constant for OLS
X_const = sm.add_constant(X)

# Run OLS
model = sm.OLS(y, X_const).fit()

print("--- Regression Results (Volume vs EFQ) ---")
print(model.summary())

print("\n--- Correlation ---")
corr, p_val = stats.pearsonr(df['Vol_Bn'], df['Mean EFQ'])
print(f"Pearson Correlation: {corr:.4f}")
print(f"P-Value: {p_val:.4f}")

# Check with Log Volume (Diminishing returns?)
df['Log_Vol'] = np.log(df['Vol_Bn'])
X_log = sm.add_constant(df['Log_Vol'])
model_log = sm.OLS(y, X_log).fit()

print("\n--- Regression Results (Log(Volume) vs EFQ) ---")
print(f"R-squared: {model_log.rsquared:.4f}")
print(f"P-value (Slope): {model_log.pvalues[1]:.4f}")
print(f"Slope: {model_log.params[1]:.4f}")

# Print data sorted by size for visual check
print("\n--- Data Sorted by Volume ---")
print(df[['marketCenterName', 'Vol_Bn', 'Mean EFQ']].sort_values('Vol_Bn'))
