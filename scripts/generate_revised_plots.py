import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

output_dir = "figures"

def save_plot(filename):
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

# Load Data
df8 = pd.read_csv('table_8_non_index_results.csv')
df4 = pd.read_csv('table_4_price_impact.csv')
df3 = pd.read_csv('table_3_order_size.csv')
df2 = pd.read_csv('table_2_volatility.csv')

# Merge for main scatter
df_main = pd.merge(df8, df4[['marketCenterName', 'Price Impact']], on='marketCenterName')

short_names = {
    'Citadel': 'Citadel',
    'G1X Securities Susquehanna': 'G1X',
    'Virtu Financial': 'Virtu',
    'Jane Street': 'Jane St',
    'UBS Securities': 'UBS',
    'Two Sigma Securities': 'Two Sigma'
}
df_main['ShortName'] = df_main['marketCenterName'].map(short_names)
df_main['EFQ_Percent'] = df_main['Mean EFQ'] * 100
df_main['Cancel_Percent'] = df_main['Cancellation Rate (%)']

# --- Figure 1: Risk vs Reward (Strategic Bifurcation) ---
# X: Log Price Impact (to handle the massive outlier G1X), Y: EFQ
# Actually, let's use a Bar Chart for Price Impact to show the scale difference, then a Scatter for EFQ.
# Or better: A Scatter plot with LOG Scale on X axis.

fig, ax = plt.subplots(figsize=(10, 6))
# Handle negative/zero impacts? No, G1X is 1.24, others ~0.01. All positive.
# df_main['Log_Price_Impact'] = np.log10(df_main['Price Impact'])

# Use Linear Scale now that data is corrected
sns.scatterplot(data=df_main, x='Price Impact', y='EFQ_Percent', s=300, hue='ShortName', palette='deep', legend=False, ax=ax)

# Add labels
for i, row in df_main.iterrows():
    ax.text(row['Price Impact']+0.0002, row['EFQ_Percent'], row['ShortName'], fontsize=12, fontweight='bold')

ax.set_title("Figure 1: Execution Efficiency (Risk vs. Reward)", fontweight='bold')
ax.set_xlabel("Price Impact ($/share) [Proxy for Flow Toxicity]")
ax.set_ylabel("EFQ Savings (%)")
ax.grid(True, linestyle='--', alpha=0.5)

# Annotate clusters
# ax.axvline(x=-1, color='gray', linestyle='--')
# ax.text(-1.5, 45, "Risk Avoiders\n(Citadel, Virtu)", fontsize=10, color='gray')
# ax.text(0, 48, "Risk Absorber\n(G1X)", fontsize=10, color='gray')

save_plot("fig1_risk_reward.png")


# --- Figure 2: The Selectivity Spectrum (Cancellations) ---
# Simple Bar Chart sorted
fig, ax = plt.subplots(figsize=(10, 6))
df_sorted = df_main.sort_values('Cancel_Percent', ascending=True)

sns.barplot(data=df_sorted, x='Cancel_Percent', y='ShortName', palette='Reds', ax=ax)

ax.set_title("Figure 2: The Selectivity Spectrum (Cancellation Rates)", fontweight='bold')
ax.set_xlabel("Cancellation Rate (%)")
ax.set_ylabel("")

save_plot("fig2_selectivity.png")


# --- Figure 3: Order Size (G1X vs Citadel) ---
# Same as before, very robust finding
df_size_melt = df3.melt(id_vars='marketCenterName', var_name='Bucket', value_name='EFQ')
df_size_melt['ShortName'] = df_size_melt['marketCenterName'].map(short_names)
df_size_melt['EFQ_Percent'] = df_size_melt['EFQ'] * 100
focus = df_size_melt[df_size_melt['ShortName'].isin(['Citadel', 'G1X'])]

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=focus, x='Bucket', y='EFQ_Percent', hue='ShortName', style='ShortName', markers=True, linewidth=3, palette=['black', 'blue'], ax=ax)
ax.set_title("Figure 3: Order Size Efficiency", fontweight='bold')
ax.set_xlabel("Order Size")
ax.set_ylabel("EFQ Savings (%)")
save_plot("fig3_order_size_v2.png")
