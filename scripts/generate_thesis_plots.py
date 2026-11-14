import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for academic publication (clean, high contrast)
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Output directory (Use local figures folder)
output_dir = r"figures"

def save_plot(filename):
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

# Load Data
try:
    df_scale = pd.read_csv('table_8_non_index_results.csv')
    df_cancel = pd.read_csv('table_6_fairness_speed.csv')
    df_size = pd.read_csv('table_3_order_size.csv')
    df_time = pd.read_csv('table_2_volatility.csv')
except FileNotFoundError as e:
    print(f"Error loading CSVs: {e}")
    exit()

# Filter for main wholesalers of interest
main_wholesalers = ['Citadel', 'G1X Securities Susquehanna', 'Virtu Financial', 'Jane Street', 'Two Sigma Securities']
short_names = {
    'Citadel': 'Citadel',
    'G1X Securities Susquehanna': 'G1X',
    'Virtu Financial': 'Virtu',
    'Jane Street': 'Jane St',
    'Two Sigma Securities': 'Two Sigma'
}

# --- Figure 1: The Scale Inversion (Bubble Plot / Twin Axis) ---
# We want to show Volume (Bar) vs EFQ (Line/Point)
# Or simpler: Scatter plot of Volume vs EFQ with labels
fig, ax = plt.subplots(figsize=(10, 6))

df_scale['ShortName'] = df_scale['marketCenterName'].map(short_names)
df_scale['Volume_Bn'] = df_scale['Total Executed Shares'] / 1e9
df_scale['EFQ_Percent'] = df_scale['Mean EFQ'] * 100

# Scatter plot
sns.scatterplot(data=df_scale, x='Volume_Bn', y='EFQ_Percent', s=200, color='0.3', ax=ax)

# Add labels
for i, row in df_scale.iterrows():
    ax.text(row['Volume_Bn']+2, row['EFQ_Percent'], row['ShortName'], fontsize=11, fontweight='bold')

# Trend line (Linear Regression)
sns.regplot(data=df_scale, x='Volume_Bn', y='EFQ_Percent', scatter=False, color='red', line_kws={'linestyle':'--', 'alpha':0.5}, ci=None, ax=ax)

# ax.set_title("Figure 4.2: The Scale Inversion\n(Market Share vs. Execution Quality)", fontweight='bold')
ax.set_xlabel("Executed Volume (Billions of Shares)")
ax.set_ylabel("Effective Spread Quotient (EFQ %)")
ax.grid(True, linestyle='--', alpha=0.7)

# Add text box for correlation
from scipy import stats
corr, _ = stats.pearsonr(df_scale['Volume_Bn'], df_scale['EFQ_Percent'])
ax.text(0.05, 0.95, f'Correlation (r): {corr:.2f}', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

save_plot("fig4_2_scale_inversion_reg.png")


# --- Figure 1: Risk vs Reward (Adverse Selection vs EFQ) ---
# Scatter plot of Price Impact vs EFQ
fig, ax = plt.subplots(figsize=(10, 6))

df_impact = pd.read_csv('table_4_price_impact.csv')
# Merge impact into df_scale
df_risk = pd.merge(df_scale, df_impact[['marketCenterName', 'Price Impact', 'Mean Realized Spread']], on='marketCenterName', how='left')

# Scatter
sns.scatterplot(data=df_risk, x='Price Impact', y='Mean Realized Spread', s=200, color='0.3', ax=ax)

# Add labels
for i, row in df_risk.iterrows():
    # Offset labels slightly
    ax.text(row['Price Impact'], row['Mean Realized Spread']+0.0002, row['ShortName'], fontsize=11, fontweight='bold')

# ax.set_title("Figure 4.4: Execution Efficiency (Risk vs. Reward)", fontweight='bold')
ax.set_xlabel("Price Impact ($/share)")
ax.set_ylabel("Realized Spread ($/share)")
ax.grid(True, linestyle='--', alpha=0.7)

save_plot("fig1_risk_reward.png")


# --- Figure 2: Fairness Gap (Cancellation Rates) ---
fig, ax = plt.subplots(figsize=(10, 6))
df_cancel['ShortName'] = df_cancel['marketCenterName'].map(short_names)
# Sort by Cancellation Rate
df_cancel = df_cancel.sort_values('Cancellation Rate (%)', ascending=False)

sns.barplot(data=df_cancel, x='Cancellation Rate (%)', y='ShortName', palette='Greys_r', ax=ax)

# ax.set_title("Figure 2: The Fairness Gap\n(Cancellation Rates by Wholesaler)", fontweight='bold')
ax.set_xlabel("Cancellation Rate (%)")
ax.set_ylabel("")
# ax.axvline(x=5, color='red', linestyle='--', label='Unconditional Liquidity Threshold')
# ax.legend()

save_plot("fig2_selectivity.png")


# --- Figure 3: Order Size Efficiency ---
# Reshape data for plotting
df_size_melt = df_size.melt(id_vars='marketCenterName', var_name='Bucket', value_name='EFQ')
df_size_melt['ShortName'] = df_size_melt['marketCenterName'].map(short_names)
df_size_melt['EFQ_Percent'] = df_size_melt['EFQ'] * 100

# Focus on G1X vs Citadel vs Average
focus_firms = ['Citadel', 'G1X']
df_focus = df_size_melt[df_size_melt['ShortName'].isin(focus_firms)].copy()

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_focus, x='Bucket', y='EFQ_Percent', hue='ShortName', style='ShortName', markers=True, dashes=False, linewidth=2.5, palette=['black', 'blue'], ax=ax)

# ax.set_title("Figure 3: Order Size Efficiency\n(Retail vs Institutional Flow)", fontweight='bold')
ax.set_xlabel("Order Size Bucket (Shares)")
ax.set_ylabel("EFQ Savings (%)")
ax.legend(title=None)

save_plot("fig5_2_order_size.png")


# --- Figure 4: Temporal Stability (G1X vs Citadel) ---
df_time['date_dt'] = pd.to_datetime(df_time['Month'])
df_time['ShortName'] = df_time['marketCenterName'].map(short_names)
df_time['EFQ_Percent'] = df_time['Mean EFQ'] * 100

# Filter for just Citadel and G1X for clean comparison
df_time_focus = df_time[df_time['ShortName'].isin(['Citadel', 'G1X'])]

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_time_focus, x='date_dt', y='EFQ_Percent', hue='ShortName', style='ShortName', linewidth=2, palette=['black', 'blue'], ax=ax)

# Highlight May 2022
ax.axvspan(pd.Timestamp('2022-06-01'), pd.Timestamp('2022-06-30'), color='gray', alpha=0.2, label="June '22 Bear Market Entry")

# ax.set_title("Figure 4: Temporal Stability (2022-2023)", fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("EFQ Savings (%)")
ax.legend()

save_plot("fig5_1_volatility.png")
