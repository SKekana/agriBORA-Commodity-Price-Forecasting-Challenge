#!/usr/bin/env python3
"""
Create visualizations for model comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
results_path = BASE_DIR / "model_comparison_results.csv"

# Load results
results = pd.read_csv(results_path)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison: Cross-Validation Performance', fontsize=16, fontweight='bold')

# Clean up model names for display
results['Model_Short'] = results['Model'].str.replace('PR #', '').str.replace(': ', '\n')

# 1. MAE Comparison
ax1 = axes[0, 0]
bars1 = ax1.barh(results['Model_Short'], results['Avg_MAE'], 
                 color=['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(results))])
ax1.set_xlabel('Mean Absolute Error (MAE)', fontweight='bold')
ax1.set_title('Average MAE (Lower is Better)', fontweight='bold', fontsize=14)
ax1.invert_yaxis()
for i, v in enumerate(results['Avg_MAE']):
    if not np.isnan(v):
        ax1.text(v + 1, i, f'{v:.2f}', va='center', fontweight='bold')

# 2. RMSE Comparison
ax2 = axes[0, 1]
bars2 = ax2.barh(results['Model_Short'], results['Avg_RMSE'],
                 color=['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(results))])
ax2.set_xlabel('Root Mean Squared Error (RMSE)', fontweight='bold')
ax2.set_title('Average RMSE (Lower is Better)', fontweight='bold', fontsize=14)
ax2.invert_yaxis()
for i, v in enumerate(results['Avg_RMSE']):
    if not np.isnan(v):
        ax2.text(v + 1, i, f'{v:.2f}', va='center', fontweight='bold')

# 3. R² Comparison
ax3 = axes[1, 0]
# For R², we need to handle negative values
r2_values = results['Avg_R2'].fillna(-1000)
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in r2_values]
bars3 = ax3.barh(results['Model_Short'], r2_values, color=colors)
ax3.set_xlabel('R² Score', fontweight='bold')
ax3.set_title('Average R² (Higher is Better, >0 is good)', fontweight='bold', fontsize=14)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.invert_yaxis()
for i, v in enumerate(r2_values):
    if v > -999:
        ax3.text(v + (5 if v > 0 else -15), i, f'{v:.2f}', va='center', fontweight='bold')

# 4. Summary table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

# Create summary table
summary_data = []
for idx, row in results.iterrows():
    model_name = row['Model'].replace('PR #', '#').split(':')[0]
    mae = f"{row['Avg_MAE']:.2f}" if not np.isnan(row['Avg_MAE']) else "N/A"
    rmse = f"{row['Avg_RMSE']:.2f}" if not np.isnan(row['Avg_RMSE']) else "N/A"
    r2 = f"{row['Avg_R2']:.4f}" if not np.isnan(row['Avg_R2']) else "N/A"
    rank = idx + 1
    summary_data.append([rank, model_name, mae, rmse, r2])

table = ax4.table(cellText=summary_data,
                  colLabels=['Rank', 'Model', 'MAE', 'RMSE', 'R²'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.1, 0.35, 0.15, 0.15, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the header
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model
for i in range(5):
    table[(1, i)].set_facecolor('#d5f4e6')
    table[(1, i)].set_text_props(weight='bold')

ax4.set_title('Performance Summary (Ranked by RMSE)', fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(BASE_DIR / 'model_comparison_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {BASE_DIR / 'model_comparison_visualization.png'}")

# Create detailed comparison plot
fig2, ax = plt.subplots(figsize=(12, 6))

# Filter out models with NaN values for better visualization
valid_results = results[results['Avg_MAE'].notna() & (results['Avg_MAE'] < 50)]

x = np.arange(len(valid_results))
width = 0.25

# Normalize metrics for comparison
mae_norm = valid_results['Avg_MAE'] / valid_results['Avg_MAE'].max()
rmse_norm = valid_results['Avg_RMSE'] / valid_results['Avg_RMSE'].max()

bars1 = ax.bar(x - width, mae_norm, width, label='MAE (normalized)', color='#3498db')
bars2 = ax.bar(x, rmse_norm, width, label='RMSE (normalized)', color='#e74c3c')

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Normalized Score (Lower is Better)', fontweight='bold', fontsize=12)
ax.set_title('Model Performance Comparison (Normalized Metrics)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(valid_results['Model_Short'], rotation=0, ha='center')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'model_comparison_normalized.png', dpi=300, bbox_inches='tight')
print(f"✓ Normalized comparison saved to: {BASE_DIR / 'model_comparison_normalized.png'}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
