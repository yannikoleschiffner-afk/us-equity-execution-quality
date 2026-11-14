import pandas as pd
import numpy as np

# Load Data
print("Loading data...")
try:
    df = pd.read_parquet('605_data.parquet')
except FileNotFoundError:
    print("Error: 605_data.parquet not found.")
    exit()

df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m')

# Filter for May and June 2022
target_months = ['2022-05', '2022-06']
df = df[df['date'].isin(target_months)]

# Wholesalers of interest
wholesalers = ['Citadel', 'G1X Securities Susquehanna', 'Jane Street']
df = df[df['marketCenterName'].isin(wholesalers)]
df = df[df['orderType'].isin([11, 12])] # Market & Marketable Limit

# Columns
es_amt_col = 'averageEffectiveSpreadForCoveredOrders'
exec_sh_col = 'cumulativeNoOfSharesOfCoveredOrdersExecuted'

df['es_dollars'] = df[es_amt_col].fillna(0) * df[exec_sh_col]

def get_mean_spread(subset):
    total_exec = subset[exec_sh_col].sum()
    if total_exec > 0:
        return subset['es_dollars'].sum() / total_exec
    return 0

# Group by Month and Wholesaler
results = df.groupby(['date', 'marketCenterName']).apply(get_mean_spread).reset_index(name='Mean_Effective_Spread')

print("\n--- Mean Effective Spread ($) Analysis ---")
print(results)
