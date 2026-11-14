import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

# Data from table_3_order_size.csv
data = {
    'Size_Bucket': ['100-499', '500-1999', '2000-4999', '5000-9999'],
    'G1X': [79.31, 68.53, 42.41, 14.51],
    'Citadel': [61.35, 54.03, 38.65, 20.60],
    'JaneStreet': [64.51, 59.52, 45.30, 25.58]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(df['Size_Bucket'], df['G1X'], marker='o', linewidth=2.5, label='G1X Susquehanna', color='#1f77b4') # Blue
ax.plot(df['Size_Bucket'], df['Citadel'], marker='s', linewidth=2.5, label='Citadel Securities', color='#d62728') # Red
ax.plot(df['Size_Bucket'], df['JaneStreet'], marker='^', linewidth=1.5, linestyle='--', label='Jane Street', color='gray') 

# Highlight the crossover
# G1X vs Citadel crossover happens between 2000-4999 and 5000-9999
# No manual annotation needed if the lines cross clearly.

ax.set_ylabel('Execution Quality (EFQ %)', fontsize=12)
ax.set_xlabel('Order Size Bucket (Shares)', fontsize=12)
# ax.set_title('The Performance Inversion: EFQ vs. Order Size', fontsize=14, fontweight='bold') # User dislikes titles
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)

# Add value labels for the final bucket to emphasize the flip
for col in ['G1X', 'Citadel']:
    y_val = df[col].iloc[-1]
    ax.text(3, y_val + 1, f'{y_val:.1f}%', ha='center', fontweight='bold', color='black')

y_val_retail_g1x = df['G1X'].iloc[0]
ax.text(0, y_val_retail_g1x + 1, f'{y_val_retail_g1x:.1f}%', ha='center', fontweight='bold', color='#1f77b4')

plt.tight_layout()
output_path = 'figures/fig5_5_order_size_inversion.png'
print(f"Saving figure to {output_path}...")
fig.savefig(output_path, dpi=300, bbox_inches='tight')
