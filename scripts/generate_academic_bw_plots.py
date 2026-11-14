"""
Generate PowerPoint-Branded Thesis Figures
==========================================
This script recreates all thesis figures with a uniform, branded style
using the PowerPoint color scheme: #0E1886 (dark blue), white, and #E5E5E6 (light gray).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats

# =============================================================================
# POWERPOINT BRAND COLOR STYLE CONFIGURATION
# =============================================================================
plt.style.use('default')  # Reset to defaults first

# PowerPoint Brand Color Palette
PRIMARY = '#0E1886'        # Dark blue - main color
SECONDARY = '#E5E5E6'      # Light gray - secondary color  
WHITE = '#FFFFFF'          # White - background/accents
PRIMARY_LIGHT = '#4A5AC4'  # Lighter blue for variety
PRIMARY_DARK = '#0A1060'   # Darker blue for emphasis

# Font settings (clean sans-serif for presentations)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Calibri', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Line and marker settings
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6

# Axes settings
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = PRIMARY
plt.rcParams['axes.labelcolor'] = PRIMARY
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True

# Grid settings
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.color'] = SECONDARY

# Figure settings
plt.rcParams['figure.facecolor'] = WHITE
plt.rcParams['axes.facecolor'] = WHITE
plt.rcParams['savefig.facecolor'] = WHITE
plt.rcParams['savefig.edgecolor'] = WHITE
plt.rcParams['savefig.dpi'] = 600

# Text color
plt.rcParams['text.color'] = PRIMARY
plt.rcParams['xtick.color'] = PRIMARY
plt.rcParams['ytick.color'] = PRIMARY

# Line styles for distinguishing series
LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# =============================================================================
# SETUP
# =============================================================================
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_plot(filename, fig=None):
    """Save figure with tight layout and high DPI."""
    if fig is None:
        fig = plt.gcf()
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data files...")

try:
    df_scale = pd.read_csv('table_8_non_index_results.csv')
    df_cancel = pd.read_csv('table_6_fairness_speed.csv')
    df_size = pd.read_csv('table_3_order_size.csv')
    df_time = pd.read_csv('table_2_volatility.csv')
    df_impact = pd.read_csv('table_4_price_impact.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    print("Please run this script from the Bachelorarbeit directory.")
    exit(1)

# Short names mapping
short_names = {
    'Citadel': 'Citadel',
    'G1X Securities Susquehanna': 'G1X',
    'Virtu Financial': 'Virtu',
    'Jane Street': 'Jane St',
    'Two Sigma Securities': 'Two Sigma',
    'UBS Securities': 'UBS'
}

# =============================================================================
# FIGURE 1: RISK VS REWARD (SCATTER PLOT)
# Price Impact vs Realized Spread
# =============================================================================
print("\nGenerating Figure 1: Risk vs Reward...")

fig, ax = plt.subplots(figsize=(8, 5.5))

# Prepare data
df_scale['ShortName'] = df_scale['marketCenterName'].map(short_names)
df_risk = pd.merge(df_scale, df_impact[['marketCenterName', 'Price Impact', 'Mean Realized Spread']], 
                   on='marketCenterName', how='left')

# Scatter plot in primary blue
ax.scatter(df_risk['Price Impact'], df_risk['Mean Realized Spread'], 
           s=150, c=PRIMARY, marker='o', zorder=5, edgecolors=PRIMARY_DARK, linewidths=1)

# Add labels with slight offset
for i, row in df_risk.iterrows():
    offset_x = 0.00015
    offset_y = 0.0002
    ax.annotate(row['ShortName'], 
                xy=(row['Price Impact'], row['Mean Realized Spread']),
                xytext=(row['Price Impact'] + offset_x, row['Mean Realized Spread'] + offset_y),
                fontsize=10, fontweight='bold', color=PRIMARY)

ax.set_xlabel("Price Impact ($/share)")
ax.set_ylabel("Realized Spread ($/share)")
ax.grid(True, linestyle='--', alpha=0.4, color=SECONDARY)

save_plot("fig1_risk_reward.png", fig)


# =============================================================================
# FIGURE 2: SELECTIVITY SPECTRUM (HORIZONTAL BAR CHART)
# Cancellation Rates by Wholesaler
# =============================================================================
print("Generating Figure 2: Selectivity Spectrum (Cancellation Rates)...")

fig, ax = plt.subplots(figsize=(8, 5))

df_cancel['ShortName'] = df_cancel['marketCenterName'].map(short_names)
df_cancel_sorted = df_cancel.sort_values('Cancellation Rate (%)', ascending=True)

# Create gradient of blues from light to dark based on position
n_bars = len(df_cancel_sorted)
# Create color gradient from light gray to primary blue
colors = [SECONDARY if i < n_bars//2 else PRIMARY for i in range(n_bars)]
# Finer gradient
alphas = np.linspace(0.4, 1.0, n_bars)

bars = ax.barh(df_cancel_sorted['ShortName'], df_cancel_sorted['Cancellation Rate (%)'],
               color=PRIMARY, edgecolor=PRIMARY_DARK, linewidth=0.8)

# Apply alpha gradient
for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)

ax.set_xlabel("Cancellation Rate (%)")
ax.set_ylabel("")
ax.grid(True, axis='x', linestyle='--', alpha=0.4, color=SECONDARY)

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.1f}%',
                xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(3, 0), textcoords='offset points',
                ha='left', va='center', fontsize=9, color=PRIMARY)

save_plot("fig2_selectivity.png", fig)


# =============================================================================
# FIGURE 3: ORDER SIZE EFFICIENCY (LINE PLOT)
# EFQ by Order Size Bucket - G1X vs Citadel
# =============================================================================
print("Generating Figure 3: Order Size Efficiency...")

fig, ax = plt.subplots(figsize=(8, 5.5))

# Prepare data
df_size_melt = df_size.melt(id_vars='marketCenterName', var_name='Bucket', value_name='EFQ')
df_size_melt['ShortName'] = df_size_melt['marketCenterName'].map(short_names)
df_size_melt['EFQ_Percent'] = df_size_melt['EFQ'] * 100

# Focus on G1X vs Citadel
focus_firms = ['Citadel', 'G1X']
df_focus = df_size_melt[df_size_melt['ShortName'].isin(focus_firms)].copy()

# Color palette for two firms
firm_colors = [PRIMARY, SECONDARY]
firm_edge_colors = [PRIMARY_DARK, '#A0A0A0']

# Plot with different line styles and markers
for i, firm in enumerate(focus_firms):
    firm_data = df_focus[df_focus['ShortName'] == firm]
    ax.plot(firm_data['Bucket'], firm_data['EFQ_Percent'], 
            linestyle=LINE_STYLES[i], marker=MARKERS[i], 
            color=PRIMARY if i == 0 else '#606060',
            linewidth=2.5, markersize=10,
            label=firm, 
            markerfacecolor=PRIMARY if i == 0 else WHITE,
            markeredgecolor=PRIMARY if i == 0 else '#606060', 
            markeredgewidth=2)

ax.set_xlabel("Order Size Bucket (Shares)")
ax.set_ylabel("EFQ Savings (%)")
ax.legend(loc='best', frameon=True, fancybox=False, edgecolor=PRIMARY)
ax.grid(True, linestyle='--', alpha=0.4, color=SECONDARY)

save_plot("fig3_order_size.png", fig)


# =============================================================================
# FIGURE 4.2: SCALE INVERSION (SCATTER - BLACK AND WHITE)
# Market Share vs Execution Quality
# =============================================================================
print("Generating Figure 4.2: Scale Inversion...")

fig, ax = plt.subplots(figsize=(8, 5.5))

df_scale['Volume_Bn'] = df_scale['Total Executed Shares'] / 1e9
df_scale['EFQ_Percent'] = df_scale['Mean EFQ'] * 100

# Scatter plot - BLACK AND WHITE style (no correlation line)
ax.scatter(df_scale['Volume_Bn'], df_scale['EFQ_Percent'], 
           s=150, c='black', marker='o', zorder=5, edgecolors='black', linewidths=1.5)

# Add labels in black
for i, row in df_scale.iterrows():
    ax.annotate(row['ShortName'], 
                xy=(row['Volume_Bn'], row['EFQ_Percent']),
                xytext=(row['Volume_Bn'] + 5, row['EFQ_Percent'] + 0.3),
                fontsize=10, fontweight='bold', color='black')

ax.set_xlabel("Executed Volume (Billions of Shares)", color='black')
ax.set_ylabel("Effective Spread Quotient (EFQ %)", color='black')
ax.tick_params(colors='black')
ax.grid(True, linestyle='--', alpha=0.4, color='gray')

# Ensure axes are black for this figure
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

save_plot("fig4_2_scale_inversion_reg.png", fig)


# =============================================================================
# FIGURE 5.1: TEMPORAL STABILITY (TIME SERIES)
# EFQ Over Time - G1X vs Citadel
# =============================================================================
print("Generating Figure 5.1: Temporal Stability...")

fig, ax = plt.subplots(figsize=(10, 5.5))

df_time['date_dt'] = pd.to_datetime(df_time['Month'])
df_time['ShortName'] = df_time['marketCenterName'].map(short_names)
df_time['EFQ_Percent'] = df_time['Mean EFQ'] * 100

# Filter for Citadel and G1X
firms_to_plot = ['Citadel', 'G1X']
line_colors = [PRIMARY, '#606060']

for i, firm in enumerate(firms_to_plot):
    firm_data = df_time[df_time['ShortName'] == firm].sort_values('date_dt')
    ax.plot(firm_data['date_dt'], firm_data['EFQ_Percent'], 
            linestyle=LINE_STYLES[i], color=line_colors[i], linewidth=2,
            label=firm)

# Highlight June 2022 Bear Market Entry (subtle shading in secondary color)
ax.axvspan(pd.Timestamp('2022-06-01'), pd.Timestamp('2022-06-30'), 
           color=SECONDARY, alpha=0.5, label="June '22 Bear Market Entry")

ax.set_xlabel("Date")
ax.set_ylabel("EFQ Savings (%)")
ax.legend(loc='best', frameon=True, fancybox=False, edgecolor=PRIMARY)
ax.grid(True, linestyle='--', alpha=0.4, color=SECONDARY)

# Format x-axis dates
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')

save_plot("fig5_1_volatility.png", fig)


# =============================================================================
# FIGURE 5.3: VIX CORRELATION (BAR CHART)
# Correlation of VIX with EFQ by Wholesaler
# =============================================================================
print("Generating Figure 5.3: VIX Correlation...")

fig, ax = plt.subplots(figsize=(8, 5.5))

# Hardcoded correlation data
data_vix = {
    'Wholesaler': ['Citadel', 'G1X', 'Jane St', 'Virtu', 'Two Sigma'],
    'VIX_EFQ_Corr': [0.61, 0.60, 0.42, 0.20, -0.50]
}
df_vix = pd.DataFrame(data_vix)

# Create bars - positive in primary blue, negative in gray
colors = [PRIMARY if x >= 0 else SECONDARY for x in df_vix['VIX_EFQ_Corr']]

bars = ax.bar(df_vix['Wholesaler'], df_vix['VIX_EFQ_Corr'], 
              color=colors, edgecolor=PRIMARY_DARK, linewidth=0.8)

ax.axhline(0, color=PRIMARY, linewidth=1)
ax.set_ylabel("Correlation (r)")
ax.grid(True, axis='y', linestyle='--', alpha=0.4, color=SECONDARY)

# Add value labels
for bar in bars:
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    offset = 0.02 if height >= 0 else -0.02
    ax.annotate(f'{height:+.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height + offset),
                ha='center', va=va, fontsize=10, fontweight='bold', color=PRIMARY)

save_plot("fig5_3_vix_correlation.png", fig)


# =============================================================================
# FIGURE 5.4: MARKET RETURN CORRELATION (BAR CHART)
# Correlation of Absolute Market Return with EFQ
# =============================================================================
print("Generating Figure 5.4: Market Return Correlation...")

fig, ax = plt.subplots(figsize=(8, 5.5))

data_mkt = {
    'Wholesaler': ['Citadel', 'G1X', 'Jane St', 'Virtu', 'Two Sigma'],
    'Mkt_Abs_EFQ_Corr': [0.37, 0.08, -0.23, 0.15, 0.06]
}
df_mkt = pd.DataFrame(data_mkt)

# Strong correlations in primary blue, weak/negative in gray
colors = [PRIMARY if abs(x) > 0.15 else SECONDARY for x in df_mkt['Mkt_Abs_EFQ_Corr']]

bars = ax.bar(df_mkt['Wholesaler'], df_mkt['Mkt_Abs_EFQ_Corr'], 
              color=colors, edgecolor=PRIMARY_DARK, linewidth=0.8)

ax.axhline(0, color=PRIMARY, linewidth=1)
ax.set_ylabel("Correlation (r)")
ax.grid(True, axis='y', linestyle='--', alpha=0.4, color=SECONDARY)

# Add value labels
for bar in bars:
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    offset = 0.02 if height >= 0 else -0.02
    ax.annotate(f'{height:+.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height + offset),
                ha='center', va=va, fontsize=10, fontweight='bold', color=PRIMARY)

save_plot("fig5_4_mkt_correlation.png", fig)


# =============================================================================
# FIGURE 5.2: ORDER SIZE COMPARISON (LINE PLOT - ALL FIRMS)
# Similar to fig3 but showing more firms for robustness
# =============================================================================
print("Generating Figure 5.2: Order Size Comparison (Full)...")

fig, ax = plt.subplots(figsize=(9, 5.5))

# Plot all available firms with distinct styles
all_firms = df_size_melt['ShortName'].dropna().unique()

# Create alternating colors between primary and gray tones
for i, firm in enumerate(all_firms):
    firm_data = df_size_melt[df_size_melt['ShortName'] == firm]
    if len(firm_data) > 0:
        # Primary firms get blue, others get gray
        color = PRIMARY if i < 2 else '#808080'
        ax.plot(firm_data['Bucket'], firm_data['EFQ_Percent'], 
                linestyle=LINE_STYLES[i % len(LINE_STYLES)], 
                marker=MARKERS[i % len(MARKERS)], 
                color=color,
                linewidth=2 if i < 2 else 1.5,
                markersize=8,
                label=firm,
                markerfacecolor=SECONDARY if i % 2 == 1 else color,
                markeredgecolor=color, markeredgewidth=1.5)

ax.set_xlabel("Order Size Bucket (Shares)")
ax.set_ylabel("EFQ Savings (%)")
ax.legend(loc='best', frameon=True, fancybox=False, edgecolor=PRIMARY, ncol=2)
ax.grid(True, linestyle='--', alpha=0.4, color=SECONDARY)

save_plot("fig5_2_order_size.png", fig)


# =============================================================================
# COMPLETION MESSAGE
# =============================================================================
print("\n" + "="*60)
print("All PowerPoint-branded figures generated successfully!")
print("="*60)
print(f"\nColor scheme: Primary Blue ({PRIMARY}), Light Gray ({SECONDARY}), White ({WHITE})")
print(f"\nOutput directory: {os.path.abspath(output_dir)}")
print("\nGenerated files:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
