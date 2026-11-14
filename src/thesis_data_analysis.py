"""
Bachelor Thesis: Wholesaler Execution Quality Analysis
=======================================================
Data Processing, Table Generation, and Statistical Analysis

This script consolidates all data analysis for the thesis examining
execution quality differences between major U.S. equity wholesalers
using SEC Rule 605 data from 2022-2023.

Author: Yannik Schiffner
Date: February 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf


# =============================================================================
# CONFIGURATION
# =============================================================================

WHOLESALERS = [
    'Citadel',
    'G1X Securities Susquehanna',
    'Jane Street',
    'Two Sigma Securities',
    'Virtu Financial'
]

# Order size bucket mapping (SEC 605 codes)
SIZE_BUCKETS = {
    21.0: '100-499',
    22.0: '500-1999',
    23.0: '2000-4999',
    24.0: '5000-9999'
}

# Column name aliases for readability
COL_PI_AMT = 'forPriceImprovementSharesTheShareWeightedAverageImprovementAmount'
COL_PI_SHARES = 'cumulativeNumberOfSharesOfCoveredOrdersWithPriceImprovement'
COL_ES_AMT = 'averageEffectiveSpreadForCoveredOrders'
COL_RS_AMT = 'averageRealizedSpreadForCoveredOrders'
COL_EXEC_SHARES = 'cumulativeNoOfSharesOfCoveredOrdersExecuted'
COL_TOTAL_SHARES = 'cumulativeNoOfSharesOfCoveredOrders'
COL_CANCELLED = 'cumulativeNoOfSharesOfCoveredOrdersCancelled'

# Speed columns
COL_SPEED_PI = 'forPriceImprovementSharesTheShareWeightedAverageTimeUntilExecution'
COL_SPEED_QUOTE = 'forSharesExecutedAtTheQuoteTheShareWeightedAverageTimeUntilExecution'
COL_SPEED_OUTSIDE = 'forSharesExecutedOutsideTheQuoteTheShareWeightedAverageTimeUntilExecution'
COL_SH_QUOTE = 'cumulativeNoOfCoveredSharesExecutedAtTheQuote'
COL_SH_OUTSIDE = 'cumulativeNumberOfSharesExecutedOutsideTheQuote'


# =============================================================================
# DATA LOADING AND FILTERING
# =============================================================================

def load_data():
    """Load all required data files."""
    print("Loading data files...")

    try:
        df = pd.read_parquet('605_data.parquet')
        sp500 = set(pd.read_csv('sp500_holdings.csv')['Symbol'].unique())
        russell = set(pd.read_csv('russell2000_holdings.csv')['Ticker'].unique())
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure all data files are in the current directory.")
        return None, None, None

    return df, sp500, russell


def apply_filters(df):
    """
    Apply standard methodology filters:
    - Date range: 2022-01 to 2023-12
    - Order types: Market (11) and Marketable Limit (12)
    - Wholesalers: Top 5 by volume
    - Outlier removal: |Realized Spread| < $5.00
    """
    print("Applying filters (2022-2023, Order Types 11 & 12)...")

    df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m')

    # Date filter
    df = df[(df['date_dt'] >= '2022-01') & (df['date_dt'] <= '2023-12')]

    # Wholesaler filter
    df = df[df['marketCenterName'].isin(WHOLESALERS)]

    # Order type filter (Market & Marketable Limit only)
    df = df[df['orderType'].isin([11, 12])]

    # Outlier removal for realized spread
    df = df[(df[COL_RS_AMT] > -5.0) & (df[COL_RS_AMT] < 5.0)]

    print(f"Filtered data: {len(df):,} rows")
    return df


def compute_metrics(df):
    """Pre-compute dollar-weighted metrics for analysis."""
    print("Computing metrics...")

    # Price improvement dollars
    df['pi_dollars'] = (
        df[COL_PI_AMT].fillna(0) * df[COL_PI_SHARES].fillna(0)
    )

    # Effective spread dollars
    df['es_dollars'] = (
        df[COL_ES_AMT].fillna(0) * df[COL_EXEC_SHARES]
    )

    # Half quoted spread dollars (benchmark)
    df['half_qs_dollars'] = 0.5 * df['es_dollars'] + df['pi_dollars']

    # Realized spread dollars
    df['rs_dollars'] = df[COL_RS_AMT].fillna(0) * df[COL_EXEC_SHARES]

    # Speed weighting (total execution time in seconds)
    df['total_time_seconds'] = (
        df[COL_SPEED_PI].fillna(0) * df[COL_PI_SHARES].fillna(0) +
        df[COL_SPEED_QUOTE].fillna(0) * df[COL_SH_QUOTE].fillna(0) +
        df[COL_SPEED_OUTSIDE].fillna(0) * df[COL_SH_OUTSIDE].fillna(0)
    )

    return df


def get_stats(subset):
    """Calculate aggregate statistics for a data subset."""
    pi_sum = subset['pi_dollars'].sum()
    hqs_sum = subset['half_qs_dollars'].sum()
    exec_sum = subset[COL_EXEC_SHARES].sum()

    efq = pi_sum / hqs_sum if hqs_sum > 0 else 0
    cancel_rate = (
        subset[COL_CANCELLED].sum() / subset[COL_TOTAL_SHARES].sum() * 100
        if subset[COL_TOTAL_SHARES].sum() > 0 else 0
    )
    avg_speed = (
        subset['total_time_seconds'].sum() / exec_sum if exec_sum > 0 else 0
    )

    return efq, cancel_rate, exec_sum, avg_speed


# =============================================================================
# TABLE GENERATION (Tables 1-8)
# =============================================================================

def generate_all_tables(df, sp500, russell):
    """Generate all summary tables for the thesis."""
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY TABLES")
    print("=" * 60)

    # Add index classification
    df['is_sp500'] = df['symbol'].isin(sp500)
    df['is_russell'] = df['symbol'].isin(russell)
    df['is_non_index'] = ~(df['is_sp500'] | df['is_russell'])

    ni_df = df[df['is_non_index']]

    # --- Table 1: High Level Index Results ---
    print("\nGenerating Table 1: High Level Index Results...")
    t1_data = []
    for w in WHOLESALERS:
        row = {'marketCenterName': w}
        for year in [2022, 2023]:
            for idx_name, mask in [('Russell 2000', 'is_russell'),
                                   ('S&P 500', 'is_sp500')]:
                sub = df[
                    (df['marketCenterName'] == w) &
                    (df['date_dt'].dt.year == year) &
                    (df[mask])
                ]
                row[f'{year} - {idx_name} (Mean EFQ)'] = get_stats(sub)[0]
        t1_data.append(row)
    t1 = pd.DataFrame(t1_data)
    t1.to_csv('table_1_high_level.csv', index=False)

    # --- Table 2: Volatility / Time Series ---
    print("Generating Table 2: Volatility (Monthly EFQ)...")
    t2 = df.groupby(['date', 'marketCenterName']).apply(
        lambda x: pd.Series({
            'Mean EFQ': get_stats(x)[0],
            'Total Shares': get_stats(x)[2]
        }),
        include_groups=False
    ).reset_index()
    t2 = t2.rename(columns={'date': 'Month'})
    t2.to_csv('table_2_volatility.csv', index=False)

    # --- Table 3: Order Size Efficiency ---
    print("Generating Table 3: Order Size Efficiency...")
    t3_data = []
    for w in WHOLESALERS:
        row = {'marketCenterName': w}
        for s_code, s_name in SIZE_BUCKETS.items():
            sub = df[(df['marketCenterName'] == w) & (df['orderSize'] == s_code)]
            row[s_name] = get_stats(sub)[0]
        t3_data.append(row)
    t3 = pd.DataFrame(t3_data)
    t3.to_csv('table_3_order_size.csv', index=False)

    # --- Table 4: Price Impact ---
    print("Generating Table 4: Price Impact...")
    t4 = df.groupby('marketCenterName').apply(
        lambda x: pd.Series({
            'Mean Eff Spread': (
                x['es_dollars'].sum() / x[COL_EXEC_SHARES].sum()
                if x[COL_EXEC_SHARES].sum() > 0 else 0
            ),
            'Mean Realized Spread': (
                x['rs_dollars'].sum() / x[COL_EXEC_SHARES].sum()
                if x[COL_EXEC_SHARES].sum() > 0 else 0
            ),
        }),
        include_groups=False
    ).reset_index()
    t4['Price Impact'] = t4['Mean Eff Spread'] - t4['Mean Realized Spread']
    t4.to_csv('table_4_price_impact.csv', index=False)

    # --- Table 5: Consistency (Coefficient of Variation) ---
    print("Generating Table 5: Consistency...")
    t5 = t2.groupby('marketCenterName')['Mean EFQ'].apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    ).reset_index()
    t5 = t5.rename(columns={'Mean EFQ': 'CV (EFQ)'})
    t5.to_csv('table_5_consistency.csv', index=False)

    # --- Table 6: Fairness / Speed ---
    print("Generating Table 6: Fairness & Speed...")
    t6 = df.groupby('marketCenterName').apply(
        lambda x: pd.Series({
            'Cancellation Rate (%)': get_stats(x)[1],
            'Avg Speed (s)': get_stats(x)[3]
        }),
        include_groups=False
    ).reset_index()
    t6.to_csv('table_6_fairness_speed.csv', index=False)

    # --- Table 7: Validity Report ---
    print("Generating Table 7: Validity Report...")
    total_vol_per_w = df.groupby('marketCenterName')[COL_EXEC_SHARES].sum()
    ni_vol_per_w = ni_df.groupby('marketCenterName')[COL_EXEC_SHARES].sum()

    t7 = df.groupby('marketCenterName').apply(
        lambda x: pd.Series({
            'Coverage Ratio (%)': (
                x[x['total_time_seconds'] > 0][COL_EXEC_SHARES].sum() /
                x[COL_EXEC_SHARES].sum() * 100
                if x[COL_EXEC_SHARES].sum() > 0 else 0
            ),
            'Improvement Rate (%)': (
                x[x[COL_PI_SHARES] > 0][COL_EXEC_SHARES].sum() /
                x[COL_EXEC_SHARES].sum() * 100
                if x[COL_EXEC_SHARES].sum() > 0 else 0
            ),
        }),
        include_groups=False
    ).reset_index()
    t7['Survivorship Retained (%)'] = t7['marketCenterName'].map(
        ni_vol_per_w / total_vol_per_w * 100
    )
    t7.to_csv('table_7_validity_report.csv', index=False)

    # --- Table 8: Non-Index Results ---
    print("Generating Table 8: Non-Index Results...")
    t8 = ni_df.groupby('marketCenterName').apply(
        lambda x: pd.Series({
            'Mean EFQ': get_stats(x)[0],
            'Cancellation Rate (%)': get_stats(x)[1],
            'Total Executed Shares': get_stats(x)[2]
        }),
        include_groups=False
    ).reset_index()
    t8.to_csv('table_8_non_index_results.csv', index=False)

    print("\nAll tables generated successfully!")
    return df


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def run_mann_whitney_test(df):
    """
    Perform Mann-Whitney U Test comparing G1X vs Citadel EFQ distributions.
    Tests whether G1X provides statistically higher price improvement.
    """
    print("\n" + "=" * 60)
    print("STATISTICAL TEST: Mann-Whitney U (G1X vs Citadel)")
    print("=" * 60)

    # Filter for valid execution volume
    df_valid = df[df[COL_EXEC_SHARES] > 0].copy()

    # Compute row-level EFQ
    df_valid['AvgPI_Global'] = (
        df_valid[COL_PI_AMT].fillna(0) * df_valid[COL_PI_SHARES].fillna(0)
    ) / df_valid[COL_EXEC_SHARES]

    df_valid['AvgHalfSpread'] = (
        0.5 * df_valid[COL_ES_AMT].fillna(0) + df_valid['AvgPI_Global']
    )

    df_valid['Row_EFQ'] = np.where(
        df_valid['AvgHalfSpread'] > 0.000001,
        df_valid['AvgPI_Global'] / df_valid['AvgHalfSpread'],
        0
    )

    # Get samples
    g1x_data = df_valid[
        df_valid['marketCenterName'] == 'G1X Securities Susquehanna'
    ].dropna(subset=['Row_EFQ'])
    citadel_data = df_valid[
        df_valid['marketCenterName'] == 'Citadel'
    ].dropna(subset=['Row_EFQ'])

    group_g1x = g1x_data['Row_EFQ']
    group_citadel = citadel_data['Row_EFQ']

    print(f"N (G1X): {len(group_g1x):,}")
    print(f"N (Citadel): {len(group_citadel):,}")

    print("\nUnweighted Row-Mean EFQ:")
    print(f"  G1X:     {group_g1x.mean():.4f}")
    print(f"  Citadel: {group_citadel.mean():.4f}")

    # Perform test (Alternative: G1X > Citadel)
    try:
        stat, p_value = mannwhitneyu(
            group_g1x, group_citadel, alternative='greater'
        )

        # Calculate effect size (Rank-Biserial Correlation)
        n1, n2 = len(group_g1x), len(group_citadel)
        r_calc = (2 * stat) / (n1 * n2) - 1

        print("\nMann-Whitney U Results:")
        print(f"  U-Statistic: {stat:,.0f}")
        print(f"  P-Value: {p_value:.5e}")
        print(f"  Rank-Biserial Correlation (r): {r_calc:.4f}")

        if p_value < 0.05:
            print("  RESULT: Significant (p < 0.05)")
            print("  G1X provides statistically higher EFQ than Citadel.")
        else:
            print("  RESULT: Not Significant")

    except Exception as e:
        print(f"Statistical test failed: {e}")


def run_regression_analysis():
    """
    Run OLS regression checking Volume vs EFQ relationship.
    Tests the "Scale Inversion" hypothesis.
    """
    print("\n" + "=" * 60)
    print("REGRESSION ANALYSIS: Volume vs EFQ")
    print("=" * 60)

    try:
        df = pd.read_csv('table_8_non_index_results.csv')
    except FileNotFoundError:
        print("Error: table_8_non_index_results.csv not found.")
        print("Please run table generation first.")
        return

    df['Vol_Bn'] = df['Total Executed Shares'] / 1e9
    X = df['Vol_Bn']
    y = df['Mean EFQ']

    # Add constant for OLS
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    print("\nLinear Regression (Volume vs EFQ):")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Slope: {model.params.iloc[1]:.6f}")
    print(f"  P-value (Slope): {model.pvalues.iloc[1]:.4f}")

    # Correlation
    corr, p_val = stats.pearsonr(df['Vol_Bn'], df['Mean EFQ'])
    print(f"\nPearson Correlation: {corr:.4f} (p = {p_val:.4f})")

    # Log-volume model
    df['Log_Vol'] = np.log(df['Vol_Bn'])
    X_log = sm.add_constant(df['Log_Vol'])
    model_log = sm.OLS(y, X_log).fit()

    print("\nLog-Linear Model (Log(Volume) vs EFQ):")
    print(f"  R-squared: {model_log.rsquared:.4f}")
    print(f"  Slope: {model_log.params.iloc[1]:.4f}")
    print(f"  P-value: {model_log.pvalues.iloc[1]:.4f}")

    print("\nData sorted by volume:")
    print(df[['marketCenterName', 'Vol_Bn', 'Mean EFQ']].sort_values('Vol_Bn'))


def run_dyhrberg_replication(df, sp500):
    """
    Replicate Dyhrberg et al. analysis: test whether "Top 2" wholesalers
    (Citadel, Virtu) have different realized spreads controlling for
    stock fixed effects.
    """
    print("\n" + "=" * 60)
    print("DYHRBERG MODEL REPLICATION")
    print("=" * 60)

    # Include UBS for this analysis
    all_wholesalers = WHOLESALERS + ['UBS Securities']

    # Filter for S&P 500 stocks only
    df_sp = df[df['symbol'].isin(sp500)].copy()
    df_sp = df_sp[df_sp['marketCenterName'].isin(all_wholesalers)]
    df_sp = df_sp.dropna(subset=[COL_RS_AMT, COL_EXEC_SHARES])
    df_sp = df_sp[df_sp[COL_EXEC_SHARES] > 0]

    # Create variables
    top_two = ['Citadel', 'Virtu Financial']
    df_sp['IsTop2'] = df_sp['marketCenterName'].isin(top_two).astype(int)
    df_sp['LogVolume'] = np.log(df_sp[COL_EXEC_SHARES])
    df_sp['RealizedSpread'] = df_sp[COL_RS_AMT]

    print(f"Observations: {len(df_sp):,}")
    print("Running: RealizedSpread ~ IsTop2 + LogVolume + C(symbol)")
    print("(This may take a moment due to stock fixed effects...)")

    try:
        results = smf.ols(
            'RealizedSpread ~ IsTop2 + LogVolume + C(symbol)',
            data=df_sp
        ).fit()

        print("\nKey Results:")
        print(f"  IsTop2 Coefficient: {results.params['IsTop2']:.6f}")
        print(f"  IsTop2 P-Value: {results.pvalues['IsTop2']:.6e}")
        print(f"  R-squared: {results.rsquared:.4f}")

        if results.pvalues['IsTop2'] < 0.05:
            if results.params['IsTop2'] < 0:
                print("  Interpretation: Top 2 have LOWER realized spreads")
            else:
                print("  Interpretation: Top 2 have HIGHER realized spreads")

    except Exception as e:
        print(f"Regression failed: {e}")


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def run_correlation_analysis():
    """
    Calculate correlations between market conditions (VIX, market returns)
    and execution quality metrics.
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS: VIX & Market Returns vs EFQ")
    print("=" * 60)

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Using pre-computed correlations.")
        print_precomputed_correlations()
        return

    # Fetch market data
    print("Fetching market data from Yahoo Finance...")
    try:
        tickers = ['^GSPC', '^VIX']
        data = yf.download(tickers, start='2022-01-01', end='2023-12-31',
                           progress=False)['Close']

        monthly = pd.DataFrame()
        monthly['VIX_Avg'] = data['^VIX'].resample('ME').mean()
        spx = data['^GSPC'].resample('ME').last()
        monthly['SPX_Return'] = spx.pct_change().fillna(0)
        monthly['SPX_Abs_Return'] = monthly['SPX_Return'].abs()
        monthly.index = monthly.index.to_period('M')

    except Exception as e:
        print(f"Failed to fetch market data: {e}")
        print_precomputed_correlations()
        return

    # Load 605 data
    try:
        df_605 = pd.read_parquet('605_data.parquet')
    except FileNotFoundError:
        print("605_data.parquet not found.")
        return

    df_605['date_dt'] = pd.to_datetime(
        df_605['date'], format='%Y-%m'
    ).dt.to_period('M')
    df_605 = df_605[df_605['orderType'].isin([11, 12])]

    # Calculate metrics
    df_605['pi_dollars'] = (
        df_605[COL_PI_AMT].fillna(0) * df_605[COL_PI_SHARES].fillna(0)
    )
    df_605['es_dollars'] = (
        df_605[COL_ES_AMT].fillna(0) * df_605[COL_EXEC_SHARES]
    )
    df_605['half_qs_dollars'] = 0.5 * df_605['es_dollars'] + df_605['pi_dollars']

    def get_monthly_metrics(subset):
        total_exec = subset[COL_EXEC_SHARES].sum()
        if total_exec == 0:
            return pd.Series({'EFQ': 0, 'Mean_Spread': 0})
        efq = subset['pi_dollars'].sum() / subset['half_qs_dollars'].sum()
        mean_spread = subset['es_dollars'].sum() / total_exec
        return pd.Series({'EFQ': efq, 'Mean_Spread': mean_spread})

    grouped = df_605.groupby(
        ['date_dt', 'marketCenterName']
    ).apply(get_monthly_metrics, include_groups=False).reset_index()

    merged = pd.merge(grouped, monthly, left_on='date_dt', right_index=True)

    # Calculate correlations
    print("\nCorrelations by Wholesaler:")
    print("-" * 50)

    results = []
    for w in WHOLESALERS:
        sub = merged[merged['marketCenterName'] == w]
        if len(sub) < 5:
            continue

        corr_vix_efq = sub['VIX_Avg'].corr(sub['EFQ'])
        corr_spx_efq = sub['SPX_Abs_Return'].corr(sub['EFQ'])
        corr_vix_spread = sub['VIX_Avg'].corr(sub['Mean_Spread'])

        print(f"\n{w}:")
        print(f"  Corr(VIX, EFQ): {corr_vix_efq:+.4f}")
        print(f"  Corr(|SPX Return|, EFQ): {corr_spx_efq:+.4f}")
        print(f"  Corr(VIX, Spread): {corr_vix_spread:+.4f}")

        results.append({
            'Wholesaler': w,
            'VIX_EFQ_Corr': corr_vix_efq,
            'SPX_Abs_EFQ_Corr': corr_spx_efq,
            'VIX_Spread_Corr': corr_vix_spread
        })

    pd.DataFrame(results).to_csv('robustness_correlations.csv', index=False)
    print("\nCorrelations saved to robustness_correlations.csv")


def print_precomputed_correlations():
    """Print pre-computed correlation values from thesis analysis."""
    print("\nPre-computed Correlations (VIX vs EFQ):")
    print("-" * 40)
    correlations = {
        'Citadel': 0.61,
        'G1X': 0.60,
        'Jane Street': 0.42,
        'Virtu': 0.20,
        'Two Sigma': -0.50
    }
    for w, corr in correlations.items():
        print(f"  {w}: {corr:+.2f}")

    print("\nPre-computed Correlations (|Market Return| vs EFQ):")
    print("-" * 40)
    mkt_correlations = {
        'Citadel': 0.37,
        'G1X': 0.08,
        'Jane Street': -0.23,
        'Virtu': 0.15,
        'Two Sigma': 0.06
    }
    for w, corr in mkt_correlations.items():
        print(f"  {w}: {corr:+.2f}")


def run_cancellation_cv_correlation():
    """
    Calculate correlation between cancellation rates and EFQ consistency.
    Tests whether selective order acceptance leads to more stable execution.
    """
    print("\n" + "=" * 60)
    print("CORRELATION: Cancellation Rate vs Consistency (CV)")
    print("=" * 60)

    # Pre-computed data from thesis tables
    data = {
        'Wholesaler': ['Citadel', 'G1X', 'Jane Street', 'Two Sigma',
                       'UBS', 'Virtu'],
        'CV': [0.032, 0.027, 0.046, 0.046, 0.024, 0.034],
        'Cancellation_Rate': [35.7, 0.3, 22.5, 45.6, 1.0, 45.4]
    }

    df = pd.DataFrame(data)
    correlation, p_value = stats.pearsonr(
        df['Cancellation_Rate'], df['CV']
    )

    print(f"Pearson Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Statistically significant correlation")
    else:
        print("Result: Not statistically significant")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function - runs complete analysis pipeline."""
    print("=" * 60)
    print("BACHELOR THESIS: WHOLESALER EXECUTION QUALITY ANALYSIS")
    print("=" * 60)

    # Load data
    df, sp500, russell = load_data()
    if df is None:
        return

    # Apply filters and compute metrics
    df = apply_filters(df)
    df = compute_metrics(df)

    # Generate all tables
    df = generate_all_tables(df, sp500, russell)

    # Run statistical tests
    run_mann_whitney_test(df)
    run_regression_analysis()
    run_dyhrberg_replication(df, sp500)

    # Run correlation analyses
    run_correlation_analysis()
    run_cancellation_cv_correlation()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - table_1_high_level.csv")
    print("  - table_2_volatility.csv")
    print("  - table_3_order_size.csv")
    print("  - table_4_price_impact.csv")
    print("  - table_5_consistency.csv")
    print("  - table_6_fairness_speed.csv")
    print("  - table_7_validity_report.csv")
    print("  - table_8_non_index_results.csv")
    print("  - robustness_correlations.csv (if yfinance available)")


if __name__ == "__main__":
    main()
