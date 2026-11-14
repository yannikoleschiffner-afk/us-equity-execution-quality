import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Hardcoded Correlation Data (from analysis_vix_market.py output)
data = {
    'Wholesaler': ['Citadel', 'G1X', 'Jane St', 'Virtu', 'Two Sigma'],
    'VIX_EFQ_Corr': [0.61, 0.60, 0.42, 0.20, -0.50],
    'Mkt_Abs_EFQ_Corr': [0.37, 0.08, -0.23, 0.15, 0.06]
}
df = pd.DataFrame(data)

# Color Palette
colors_vix = ['#2ca02c' if x > 0 else '#d62728' for x in df['VIX_EFQ_Corr']]
colors_mkt = ['#1f77b4' if abs(x) < 0.15 else '#ff7f0e' for x in df['Mkt_Abs_EFQ_Corr']] 

# --- Figure 5.3: VIX Correlation ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
bars1 = ax1.bar(df['Wholesaler'], df['VIX_EFQ_Corr'], color=colors_vix, alpha=0.8)
# ax1.set_title('Correlation: VIX vs. EFQ', fontsize=14, fontweight='bold')
ax1.set_ylabel('Correlation (r)', fontsize=12)
ax1.axhline(0, color='black', linewidth=1)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:+.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path1 = 'figures/fig5_3_vix_correlation.png'
print(f"Saving figure to {output_path1}...")
fig1.savefig(output_path1, dpi=300, bbox_inches='tight')

# --- Figure 5.4: Market Return Correlation ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
bars2 = ax2.bar(df['Wholesaler'], df['Mkt_Abs_EFQ_Corr'], color=colors_mkt, alpha=0.8)
# ax2.set_title('Correlation: Abs(Market Return) vs. EFQ', fontsize=14, fontweight='bold')
ax2.set_ylabel('Correlation (r)', fontsize=12)
ax2.axhline(0, color='black', linewidth=1)
# ax2.axhline(0.1, color='gray', linestyle=':', alpha=0.5, label='Neutral Threshold')
# ax2.axhline(-0.1, color='gray', linestyle=':', alpha=0.5)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
# ax2.legend(['Zero', 'Neutral Zone (~0.1)'])

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:+.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path2 = 'figures/fig5_4_mkt_correlation.png'
print(f"Saving figure to {output_path2}...")
fig2.savefig(output_path2, dpi=300, bbox_inches='tight')

print(f"Plots saved successfully.")
