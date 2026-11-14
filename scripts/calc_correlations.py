import pandas as pd
import numpy as np
from scipy import stats

# 1. Load Data
print("Loading Parquet Data...")
try:
    df_605 = pd.read_parquet('605_data.parquet')
except FileNotFoundError:
    print("Error: 605_data.parquet not found.")
    exit()

print("Loading VIX Data...")
try:
    df_vix = pd.read_csv('vix_monthly.csv')
except FileNotFoundError:
    print("Error: vix_monthly.csv not found.")
    exit()

# 2. Process 605 Data (Aggregating Month/Wholesaler)
df_605['date_dt'] = pd.to_datetime(df_605['date'], format='%Y-%m')
# Filter for 2022-2023 and Order Types 11/12
df_605 = df_605[(df_605['date_dt'] >= '2022-01') & (df_605['date_dt'] <= '2023-12')]
df_605 = df_605[df_605['orderType'].isin([11, 12])]

# Define Metrics
pi_dollars_col = 'pi_dollars' # Need to recalculate these from raw cols if not preserved
# Recalculating helper dollars as in generate_tables
df_605['pi_dollars'] = df_605['forPriceImprovementSharesTheShareWeightedAverageImprovementAmount'].fillna(0) * df_605['cumulativeNumberOfSharesOfCoveredOrdersWithPriceImprovement'].fillna(0)
df_605['es_dollars'] = df_605['averageEffectiveSpreadForCoveredOrders'].fillna(0) * df_605['cumulativeNoOfSharesOfCoveredOrdersExecuted']
df_605['half_qs_dollars'] = 0.5 * df_605['es_dollars'] + df_605['pi_dollars']

def get_monthly_metrics(subset):
    total_exec = subset['cumulativeNoOfSharesOfCoveredOrdersExecuted'].sum()
    if total_exec == 0: return 0, 0
    
    efq = subset['pi_dollars'].sum() / subset['half_qs_dollars'].sum()
    mean_spread = subset['es_dollars'].sum() / total_exec
    return pd.Series({'EFQ': efq, 'Mean_Spread': mean_spread})

# Focus on G1X and Citadel
focus_wholesalers = ['Citadel', 'G1X Securities Susquehanna']
df_focus = df_605[df_605['marketCenterName'].isin(focus_wholesalers)]

monthly_stats = df_focus.groupby(['date', 'marketCenterName']).apply(get_monthly_metrics).reset_index()

# 3. Merge with VIX
df_vix = df_vix.rename(columns={'Month': 'date'})
merged = pd.merge(monthly_stats, df_vix, on='date', how='inner')

# 4. Calculate Correlations
print("\n--- CORRELATION MATRIX (Pearson r) ---")
for w in focus_wholesalers:
    sub = merged[merged['marketCenterName'] == w]
    if len(sub) < 5: continue
    
    # Correlation VIX vs EFQ
    corr_efq, _ = stats.pearsonr(sub['VIX_Avg'], sub['EFQ'])
    # Correlation VIX vs Spread
    corr_spread, _ = stats.pearsonr(sub['VIX_Avg'], sub['Mean_Spread'])
    
    print(f"\nWholesaler: {w}")
    print(f"  Corr(VIX, EFQ): {corr_efq:.4f}  (Expected: Negative?)")
    print(f"  Corr(VIX, Spread): {corr_spread:.4f}  (Expected: Positive)")

# Print Data for Inspection
print("\n--- Monthly Data Snapshot ---")
print(merged[['date', 'marketCenterName', 'VIX_Avg', 'EFQ', 'Mean_Spread']].head(10))
