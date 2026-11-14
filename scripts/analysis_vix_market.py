import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats

# 1. Fetch Market Data (Source of Truth)
print("Fetching Market Data (yfinance)...")
tickers = ['^GSPC', '^VIX']
data = yf.download(tickers, start='2022-01-01', end='2023-12-31')['Close']

# Resample to Monthly
monthly = pd.DataFrame()
monthly['VIX_Avg'] = data['^VIX'].resample('ME').mean()
# S&P 500 Returns
spx = data['^GSPC'].resample('ME').last()
monthly['SPX_Return'] = spx.pct_change().fillna(0)
monthly['SPX_Abs_Return'] = monthly['SPX_Return'].abs()
monthly.index = monthly.index.to_period('M')

print(monthly.head())

# 2. Load 605 Data
print("\nLoading 605 Parquet...")
try:
    df_605 = pd.read_parquet('605_data.parquet')
except FileNotFoundError:
    print("Error: 605_data.parquet not found.")
    exit()

df_605['date_dt'] = pd.to_datetime(df_605['date'], format='%Y-%m').dt.to_period('M')
# df_605 = df_605[df_605['marketCenterName'].isin(['Citadel', 'G1X Securities Susquehanna'])] # Removed to analyze all
df_605 = df_605[df_605['orderType'].isin([11, 12])]

# Calculate Metrics
df_605['pi_dollars'] = df_605['forPriceImprovementSharesTheShareWeightedAverageImprovementAmount'].fillna(0) * df_605['cumulativeNumberOfSharesOfCoveredOrdersWithPriceImprovement'].fillna(0)
df_605['es_dollars'] = df_605['averageEffectiveSpreadForCoveredOrders'].fillna(0) * df_605['cumulativeNoOfSharesOfCoveredOrdersExecuted']
df_605['half_qs_dollars'] = 0.5 * df_605['es_dollars'] + df_605['pi_dollars']

def get_stats(subset):
    total_exec = subset['cumulativeNoOfSharesOfCoveredOrdersExecuted'].sum()
    if total_exec == 0: return pd.Series({'EFQ': 0, 'Mean_Spread': 0})
    efq = subset['pi_dollars'].sum() / subset['half_qs_dollars'].sum()
    mean_spread = subset['es_dollars'].sum() / total_exec
    return pd.Series({'EFQ': efq, 'Mean_Spread': mean_spread})

grouped = df_605.groupby(['date_dt', 'marketCenterName']).apply(get_stats).reset_index()

# 3. Merge
merged = pd.merge(grouped, monthly, left_on='date_dt', right_index=True)

# 4. Analyze Correlations
wholesalers = ['Citadel', 'G1X Securities Susquehanna', 'Jane Street', 'Two Sigma Securities', 'Virtu Financial']
results = []

print("\n--- ROBUSTNESS CHECK: CORRELATIONS ---")
for w in wholesalers:
    sub = merged[merged['marketCenterName'] == w]
    
    # Pearson Correlations
    corr_vix_efq = sub['VIX_Avg'].corr(sub['EFQ'])
    corr_spx_efq = sub['SPX_Abs_Return'].corr(sub['EFQ'])
    corr_vix_spread = sub['VIX_Avg'].corr(sub['Mean_Spread'])
    
    print(f"\n{w}:")
    print(f"  Corr(VIX, EFQ): {corr_vix_efq:.4f}")
    print(f"  Corr(Abs(SPX_Return), EFQ): {corr_spx_efq:.4f}")
    print(f"  Corr(VIX, Spread): {corr_vix_spread:.4f}") # Confirming Mechanism
    
    results.append({
        'Wholesaler': w,
        'VIX_EFQ_Corr': corr_vix_efq,
        'SPX_Abs_EFQ_Corr': corr_spx_efq,
        'VIX_Spread_Corr': corr_vix_spread
    })

# 5. Export for Artifact
pd.DataFrame(results).to_csv('robustness_correlations.csv', index=False)
merged.to_csv('robustness_merged_data.csv', index=False)
print("\nData exported to robustness_merged_data.csv")
