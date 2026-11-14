import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def run_analysis():
    print("Loading data...")
    # 1. Load Data
    try:
        df = pd.read_parquet('605_data.parquet')
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return

    # 2. Filter Setup
    wholesalers = ['Citadel', 'G1X Securities Susquehanna', 'Jane Street', 
                   'Two Sigma Securities', 'Virtu Financial']
    
    # Filter by Date (2022-2023) as per methodology
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m')
    df = df[(df['date_dt'] >= '2022-01') & (df['date_dt'] <= '2023-12')]

    # Filter by Wholesaler and Order Type
    df = df[df['marketCenterName'].isin(wholesalers)]
    df = df[df['orderType'].isin([11, 12])] # Market & Marketable Limit
    
    # Filter for valid execution volume to avoid division by zero
    df = df[df['cumulativeNoOfSharesOfCoveredOrdersExecuted'] > 0]

    print(f"Data loaded and filtered. Rows: {len(df)}")

    # 3. Metric Calculation (Corrected Row-Level Logic)
    
    # Column Aliases
    col_es = 'averageEffectiveSpreadForCoveredOrders'
    col_pi_amt = 'forPriceImprovementSharesTheShareWeightedAverageImprovementAmount'
    col_pi_sh = 'cumulativeNumberOfSharesOfCoveredOrdersWithPriceImprovement'
    col_exec_sh = 'cumulativeNoOfSharesOfCoveredOrdersExecuted'

    # Handle NaNs
    df[col_pi_amt] = df[col_pi_amt].fillna(0)
    df[col_pi_sh] = df[col_pi_sh].fillna(0)
    df[col_es] = df[col_es].fillna(0)

    # Calculate Global Average Price Improvement (Spread Savings) per executed share
    # AvgPI_Global = (AvgPI_per_improved * ImprovedShares) / TotalExecuted
    df['AvgPI_Global'] = (df[col_pi_amt] * df[col_pi_sh]) / df[col_exec_sh]

    # Calculate Half Quoted Spread (The Benchmark)
    # QS_Global = ES + 2 * AvgPI_Global
    # Half_QS = QS_Global / 2 = 0.5 * ES + AvgPI_Global
    df['AvgHalfSpread'] = 0.5 * df[col_es] + df['AvgPI_Global']

    # Calculate Row-Level EFQ (Previously PIC) - Savings %
    # EFQ = AvgPI_Global / AvgHalfSpread
    # If HalfSpread is 0, EFQ is undefined (or 0).
    df['Row_EFQ'] = np.where(df['AvgHalfSpread'] > 0.000001, 
                             df['AvgPI_Global'] / df['AvgHalfSpread'], 
                             0)
    
    # 4. Perform Mann-Whitney U Test: G1X vs Citadel
    print("\n--- Statistical Test: G1X vs Citadel ---")

    g1x_data = df[df['marketCenterName'] == 'G1X Securities Susquehanna'].dropna(subset=['Row_EFQ'])
    citadel_data = df[df['marketCenterName'] == 'Citadel'].dropna(subset=['Row_EFQ'])

    group_g1x = g1x_data['Row_EFQ']
    group_citadel = citadel_data['Row_EFQ']

    print(f"N (G1X): {len(group_g1x)}")
    print(f"N (Citadel): {len(group_citadel)}")
    
    # Calculate Unweighted Mean (Distribution Mean)
    print(f"\nUnweighted Row-Mean EFQ (Used in Test):")
    print(f"G1X:     {group_g1x.mean():.4f}")
    print(f"Citadel: {group_citadel.mean():.4f}")

    # Calculate Volume-Weighted Mean (Matches Table 4.1)
    # Weighted Mean = Sum(Row_EFQ * Shares) / Sum(Shares)
    # Note: Accurately, Aggregate EFQ is Sum(PI)/Sum(HalfSpread), but Weighted Mean of Row EFQ is a good approx.
    # Let's use the Aggregate Ratio Calculation for true validation:
    
    def calc_aggregate_efq(d):
        total_pi = (d['AvgPI_Global'] * d['cumulativeNoOfSharesOfCoveredOrdersExecuted']).sum()
        total_cost = (d['AvgHalfSpread'] * d['cumulativeNoOfSharesOfCoveredOrdersExecuted']).sum()
        return total_pi / total_cost if total_cost > 0 else 0

    wt_g1x = calc_aggregate_efq(g1x_data)
    wt_citadel = calc_aggregate_efq(citadel_data)

    print(f"\nVolume-Weighted Aggregate EFQ (Matches Summary Tables):")
    print(f"G1X:     {wt_g1x:.4f}")
    print(f"Citadel: {wt_citadel:.4f}")

    # Run Test (Alternative 'greater' -> G1X > Citadel)
    # We test the distribution of *buckets* (rows), which checks consistency of strategy across all symbols.
    try:
        stat, p_value = mannwhitneyu(group_g1x, group_citadel, alternative='greater')
        
        # Calculate Rank-Biserial Correlation (Effect Size)
        n1 = len(group_g1x)
        n2 = len(group_citadel)
        u1 = stat
        r_calc = (2 * u1) / (n1 * n2) - 1

        print(f"\nMann-Whitney U Result (on Unweighted Rows):")
        print(f"Statistic: {stat}")
        print(f"P-Value: {p_value:.5e}")
        print(f"Rank-Biserial Correlation (r): {r_calc:.4f}")

        if p_value < 0.05:
            print("RESULT: Significant. G1X provides statistically higher savings than Citadel.")
        else:
            print("RESULT: Not Significant.")
            
    except Exception as e:
        print(f"Statistical test failed: {e}")

if __name__ == "__main__":
    run_analysis()
