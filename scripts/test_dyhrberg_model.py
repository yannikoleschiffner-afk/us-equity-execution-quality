import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def run_dyhrberg_replication():
    print("Loading data...")
    # 1. Load Data
    try:
        df = pd.read_parquet('605_data.parquet')
        sp500 = set(pd.read_csv('sp500_holdings.csv')['Symbol'].unique())
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Filter Setup
    print("Filtering for S&P 500 and Top wholesalers...")
    wholesalers = ['Citadel', 'G1X Securities Susquehanna', 'Jane Street', 
                   'Two Sigma Securities', 'UBS Securities', 'Virtu Financial']
    
    # Filter for Wholesalers
    df = df[df['marketCenterName'].isin(wholesalers)]
    
    # Filter for Market/Marketable Limit (11, 12)
    df = df[df['orderType'].isin([11, 12])]

    # Filter for S&P 500 Only (To use Stock Fixed Effects effectively)
    df = df[df['symbol'].isin(sp500)]
    
    # Filter for valid data
    df = df.dropna(subset=['averageRealizedSpreadForCoveredOrders', 'cumulativeNoOfSharesOfCoveredOrdersExecuted'])
    df = df[df['cumulativeNoOfSharesOfCoveredOrdersExecuted'] > 0]

    # 3. Create Dyhrberg Variables
    # "Top Two" Dummy: Citadel & Virtu are the largest
    top_two = ['Citadel', 'Virtu Financial']
    df['IsTop2'] = df['marketCenterName'].isin(top_two).astype(int)
    
    # Log Volume (Control variable)
    df['LogVolume'] = np.log(df['cumulativeNoOfSharesOfCoveredOrdersExecuted'])
    
    # Dependent Variable: Realized Spread ($ Amount per share)
    # Note: We use Dollar Amount because we don't have Price to calculate BPS.
    # Fixed Effects will handle the varying price levels between stocks.
    df['RealizedSpread'] = df['averageRealizedSpreadForCoveredOrders']

    print(f"Data Ready. Observations: {len(df)}")
    print("Running OLS Regression: RealizedSpread ~ IsTop2 + LogVolume + C(Symbol)...")
    print("Note: This may take a moment due to the large number of dummy variables for S&P 500 stocks.")

    # 4. Run Regression (Using statsmodels formulas)
    # We use a simplified model first without full fixed effects to ensure it runs quickly, 
    # then we can try adding them if memory permits.
    # Model A: Simple (No FE) - just to check raw correlation
    # Model B: With Fixed Effects (The robust check)
    
    try:
        # Using "C(symbol)" creates ~500 dummy variables. This might be heavy.
        # Let's try to model it efficiently.
        
        # Alternative: Demean the data (Absorb Fixed Effects manually) to save memory?
        # For now, let's try the direct approach on a subset if needed, or full if powerful enough.
        # Let's just use 2023 data to keep it lighter?
        # df = df[df['date'] >= '202301'] 
        
        results = smf.ols('RealizedSpread ~ IsTop2 + LogVolume + C(symbol)', data=df).fit()
        
        print("\n--- Regression Results (Dependent Var: Realized Spread $) ---")
        print(results.summary().tables[1]) # Print just the coefficients table
        print("\nKey Interpretation:")
        print(f"IsTop2 Coefficient: {results.params['IsTop2']:.6f}")
        print(f"P-Value: {results.pvalues['IsTop2']:.6e}")
        
    except Exception as e:
        print(f"Regression failed: {e}")

if __name__ == "__main__":
    run_dyhrberg_replication()
