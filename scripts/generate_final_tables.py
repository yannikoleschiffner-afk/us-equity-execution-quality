import pandas as pd
import numpy as np

# 1. Load Data and Helpers
print("Loading data...")
df = pd.read_parquet('605_data.parquet')
df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m')

sp500 = set(pd.read_csv('sp500_holdings.csv')['Symbol'].unique())
russell = set(pd.read_csv('russell2000_holdings.csv')['Ticker'].unique())

wholesalers = ['Citadel', 'G1X Securities Susquehanna', 'Jane Street', 'Two Sigma Securities', 'Virtu Financial']

# 2. Apply Global Filters
print("Applying filters (2022-2023, Order Types 11 & 12)...")
df = df[(df['date_dt'] >= '2022-01') & (df['date_dt'] <= '2023-12')]
df = df[df['marketCenterName'].isin(wholesalers)]
# CRITICAL FIX: Only Market (11) and Marketable Limit (12)
df = df[df['orderType'].isin([11, 12])]

# CRITICAL FIX: Filter Data Errors (Outliers)
# Some rows have Realized Spreads < -1000 (e.g. AMC data errors for G1X)
# We filter reasonably: |Realized Spread| < $5.00 per share
rs_col_raw = 'averageRealizedSpreadForCoveredOrders'
df = df[(df[rs_col_raw] > -5.0) & (df[rs_col_raw] < 5.0)]

# Add helper columns
pi_amt_col = 'forPriceImprovementSharesTheShareWeightedAverageImprovementAmount'
pi_sh_col = 'cumulativeNumberOfSharesOfCoveredOrdersWithPriceImprovement'
es_amt_col = 'averageEffectiveSpreadForCoveredOrders'
rs_amt_col = 'averageRealizedSpreadForCoveredOrders'
exec_sh_col = 'cumulativeNoOfSharesOfCoveredOrdersExecuted'
total_sh_col = 'cumulativeNoOfSharesOfCoveredOrders'
cancelled_sh_col = 'cumulativeNoOfSharesOfCoveredOrdersCancelled'

# Speed columns
speed_pi = 'forPriceImprovementSharesTheShareWeightedAverageTimeUntilExecution'
speed_quote = 'forSharesExecutedAtTheQuoteTheShareWeightedAverageTimeUntilExecution'
speed_outside = 'forSharesExecutedOutsideTheQuoteTheShareWeightedAverageTimeUntilExecution'
sh_quote = 'cumulativeNoOfCoveredSharesExecutedAtTheQuote'
sh_outside = 'cumulativeNumberOfSharesExecutedOutsideTheQuote'

# Pre-calculate components
df['pi_dollars'] = df[pi_amt_col].fillna(0) * df[pi_sh_col].fillna(0)
df['es_dollars'] = df[es_amt_col].fillna(0) * df[exec_sh_col]
# Half Quoted Spread Dollars = 0.5 * ES * Exec_shares + PI_amt * PI_shares
df['half_qs_dollars'] = 0.5 * df['es_dollars'] + df['pi_dollars']

df['rs_dollars'] = df[rs_amt_col].fillna(0) * df[exec_sh_col]

# Speed Weight (Total seconds for all executed shares)
# total_time = time_pi * sh_pi + time_quote * sh_quote + time_outside * sh_outside
df['total_time_seconds'] = (
    df[speed_pi].fillna(0) * df[pi_sh_col].fillna(0) +
    df[speed_quote].fillna(0) * df[sh_quote].fillna(0) +
    df[speed_outside].fillna(0) * df[sh_outside].fillna(0)
)

# Identify Index and Non-Index
df['is_sp500'] = df['symbol'].isin(sp500)
df['is_russell'] = df['symbol'].isin(russell)
df['is_non_index'] = ~(df['is_sp500'] | df['is_russell'])

def get_stats(subset):
    pi_sum = subset['pi_dollars'].sum()
    hqs_sum = subset['half_qs_dollars'].sum()
    exec_sum = subset[exec_sh_col].sum()
    
    efq = pi_sum / hqs_sum if hqs_sum > 0 else 0
    cancel_rate = (subset[cancelled_sh_col].sum() / subset[total_sh_col].sum() * 100) if subset[total_sh_col].sum() > 0 else 0
    avg_speed = subset['total_time_seconds'].sum() / exec_sum if exec_sum > 0 else 0
    
    return efq, cancel_rate, exec_sum, avg_speed

# --- Table 8: Non-Index Results ---
print("Generating Table 8...")
ni_df = df[df['is_non_index']]
t8 = ni_df.groupby('marketCenterName').apply(lambda x: pd.Series({
    'Mean EFQ': get_stats(x)[0],
    'Cancellation Rate (%)': get_stats(x)[1],
    'Total Executed Shares': get_stats(x)[2]
}), include_groups=False).reset_index()
t8.to_csv('table_8_non_index_results.csv', index=False)

# --- Table 1: High Level ---
print("Generating Table 1...")
t1_data = []
for w in wholesalers:
    row = {'marketCenterName': w}
    for year in [2022, 2023]:
        for idx_name, mask in [('Russell 2000', 'is_russell'), ('S&P 500', 'is_sp500')]:
            sub = df[(df['marketCenterName'] == w) & (df['date_dt'].dt.year == year) & (df[mask])]
            row[f'{year} - {idx_name} (Mean EFQ)'] = get_stats(sub)[0]
    t1_data.append(row)
t1 = pd.DataFrame(t1_data)
t1.to_csv('table_1_high_level.csv', index=False)

# --- Table 2: Volatility ---
print("Generating Table 2...")
t2 = df.groupby(['date', 'marketCenterName']).apply(lambda x: pd.Series({
    'Mean EFQ': get_stats(x)[0],
    'Total Shares': get_stats(x)[2]
}), include_groups=False).reset_index()
t2 = t2.rename(columns={'date': 'Month'})
t2.to_csv('table_2_volatility.csv', index=False)

# --- Table 3: Order Size Efficiency ---
print("Generating Table 3...")
size_map = {21.0: '100-499', 22.0: '500-1999', 23.0: '2000-4999', 24.0: '5000-9999'}
t3_data = []
for w in wholesalers:
    row = {'marketCenterName': w}
    for s_code, s_name in size_map.items():
        sub = df[(df['marketCenterName'] == w) & (df['orderSize'] == s_code)]
        row[s_name] = get_stats(sub)[0]
    t3_data.append(row)
t3 = pd.DataFrame(t3_data)
t3.to_csv('table_3_order_size.csv', index=False)

# --- Table 4: Price Impact ---
print("Generating Table 4...")
t4 = df.groupby('marketCenterName').apply(lambda x: pd.Series({
    'Mean Eff Spread': x['es_dollars'].sum() / x[exec_sh_col].sum() if x[exec_sh_col].sum() > 0 else 0,
    'Mean Realized Spread': x['rs_dollars'].sum() / x[exec_sh_col].sum() if x[exec_sh_col].sum() > 0 else 0,
}), include_groups=False).reset_index()
t4['Price Impact'] = t4['Mean Eff Spread'] - t4['Mean Realized Spread']
t4.to_csv('table_4_price_impact.csv', index=False)

# --- Table 5: Consistency ---
print("Generating Table 5...")
t5 = t2.groupby('marketCenterName')['Mean EFQ'].apply(lambda x: x.std() / x.mean() if x.mean() > 0 else 0).reset_index()
t5 = t5.rename(columns={'Mean EFQ': 'CV (EFQ)'})
t5.to_csv('table_5_consistency.csv', index=False)

# --- Table 6: Fairness / Speed ---
print("Generating Table 6...")
t6 = df.groupby('marketCenterName').apply(lambda x: pd.Series({
    'Cancellation Rate (%)': get_stats(x)[1],
    'Avg Speed (s)': get_stats(x)[3]
}), include_groups=False).reset_index()
t6.to_csv('table_6_fairness_speed.csv', index=False)

# --- Table 7: Validity Report ---
print("Generating Table 7...")
total_vol_per_w = df.groupby('marketCenterName')[exec_sh_col].sum()
ni_vol_per_w = ni_df.groupby('marketCenterName')[exec_sh_col].sum()

t7 = df.groupby('marketCenterName').apply(lambda x: pd.Series({
    'Coverage Ratio (%)': (x[x['total_time_seconds'] > 0][exec_sh_col].sum() / x[exec_sh_col].sum() * 100) if x[exec_sh_col].sum() > 0 else 0,
    'Improvement Rate (%)': (x[x[pi_sh_col] > 0][exec_sh_col].sum() / x[exec_sh_col].sum() * 100) if x[exec_sh_col].sum() > 0 else 0,
}), include_groups=False).reset_index()
t7['Survivorship Retained (%)'] = t7['marketCenterName'].map(ni_vol_per_w / total_vol_per_w * 100)
t7.to_csv('table_7_validity_report.csv', index=False)

print("All tables regenerated successfully!")
