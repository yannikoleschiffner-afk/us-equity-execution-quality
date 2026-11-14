import pandas as pd
import scipy.stats as stats

# Data from the csv files
data = {
    'Wholesaler': ['Citadel', 'G1X', 'Jane Street', 'Two Sigma', 'UBS', 'Virtu'],
    'CV': [0.032438920, 0.026806670, 0.046178897, 0.046480015, 0.024444480, 0.034236081],
    'Cancellation_Rate': [35.71496883, 0.34037983, 22.51759990, 45.56009222, 1.00093606, 45.39301401]
}

df = pd.DataFrame(data)

# Calculate Pearson correlation
correlation, p_value = stats.pearsonr(df['Cancellation_Rate'], df['CV'])

print(f"Correlation between Cancellation Rate and CV: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
