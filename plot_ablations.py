"""
Plot ablation study results: profit, error rate, wage parity.

Compares:
- A (baseline): No novelty weighting (weights=1)
- B (no_lambda): No constraint enforcement (λ=0)
- C (full): Full model with all features
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def read_csv(filepath):
    """Read CSV file and return dict of columns."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
    return {k: np.array(v) for k, v in data.items()}

# Read CSV data
df_A = read_csv('logs/ablation_A_baseline.csv')
df_B = read_csv('logs/ablation_B_no_lambda.csv')
df_C = read_csv('logs/ablation_C_full.csv')

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Smoothing function (simple moving average)
def smooth(data, window=20):
    """Smooth data with moving average."""
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(data[start:i+1])
    return smoothed

# Set up figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Colors and labels
colors = {'A': '#e74c3c', 'B': '#3498db', 'C': '#2ecc71'}
labels = {
    'A': 'Baseline (weights=1)',
    'B': 'No constraint (λ=0)',
    'C': 'Full model'
}

# Plot 1: Profit
ax = axes[0]
ax.plot(df_A['episode'], smooth(df_A['profit']), color=colors['A'],
        linewidth=2, label=labels['A'], alpha=0.8)
ax.plot(df_B['episode'], smooth(df_B['profit']), color=colors['B'],
        linewidth=2, label=labels['B'], alpha=0.8)
ax.plot(df_C['episode'], smooth(df_C['profit']), color=colors['C'],
        linewidth=2, label=labels['C'], alpha=0.8)
ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Profit ($/hr)', fontsize=11)
ax.set_title('Economic Profit', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(y=18, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Human wage ($18/hr)')

# Plot 2: Error Rate
ax = axes[1]
ax.plot(df_A['episode'], smooth(df_A['err_rate']), color=colors['A'],
        linewidth=2, label=labels['A'], alpha=0.8)
ax.plot(df_B['episode'], smooth(df_B['err_rate']), color=colors['B'],
        linewidth=2, label=labels['B'], alpha=0.8)
ax.plot(df_C['episode'], smooth(df_C['err_rate']), color=colors['C'],
        linewidth=2, label=labels['C'], alpha=0.8)
ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Error Rate', fontsize=11)
ax.set_title('Error Rate (SLA = 6%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.06, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (6%)')

# Plot 3: Wage Parity
ax = axes[2]
ax.plot(df_A['episode'], smooth(df_A['wage_parity']), color=colors['A'],
        linewidth=2, label=labels['A'], alpha=0.8)
ax.plot(df_B['episode'], smooth(df_B['wage_parity']), color=colors['B'],
        linewidth=2, label=labels['B'], alpha=0.8)
ax.plot(df_C['episode'], smooth(df_C['wage_parity']), color=colors['C'],
        linewidth=2, label=labels['C'], alpha=0.8)
ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Wage Parity (ŵᵣ/wₕ)', fontsize=11)
ax.set_title('Wage Parity (target = 1.0)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Parity (1.0)')

plt.tight_layout()
plt.savefig('plots/ablation_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/ablation_comparison.png")

# Create individual plots for each metric
metrics = [
    ('profit', 'Profit ($/hr)', 'Economic Profit Over Training'),
    ('err_rate', 'Error Rate', 'Error Rate Over Training (SLA = 6%)'),
    ('wage_parity', 'Wage Parity (ŵᵣ/wₕ)', 'Wage Parity Over Training')
]

for metric, ylabel, title in metrics:
    plt.figure(figsize=(8, 5))

    plt.plot(df_A['episode'], smooth(df_A[metric]), color=colors['A'],
             linewidth=2.5, label=labels['A'], alpha=0.8)
    plt.plot(df_B['episode'], smooth(df_B[metric]), color=colors['B'],
             linewidth=2.5, label=labels['B'], alpha=0.8)
    plt.plot(df_C['episode'], smooth(df_C[metric]), color=colors['C'],
             linewidth=2.5, label=labels['C'], alpha=0.8)

    # Add reference lines
    if metric == 'err_rate':
        plt.axhline(y=0.06, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='SLA Target (6%)')
    elif metric == 'wage_parity':
        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,
                   alpha=0.7, label='Parity (1.0)')
    elif metric == 'profit':
        plt.axhline(y=18, color='gray', linestyle='--', linewidth=2,
                   alpha=0.5, label='Human wage ($18/hr)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, framealpha=0.95, loc='best')
    plt.grid(True, alpha=0.3)

    filename = f'plots/ablation_{metric}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

# Print summary statistics
print("\n" + "="*60)
print("ABLATION STUDY SUMMARY (last 50 episodes)")
print("="*60)

for name, df, color in [('A (baseline)', df_A, 'A'),
                         ('B (no λ)', df_B, 'B'),
                         ('C (full)', df_C, 'C')]:
    tail_idx = -50
    print(f"\n{name}:")
    print(f"  Profit:       ${np.mean(df['profit'][tail_idx:]):.2f} ± ${np.std(df['profit'][tail_idx:]):.2f}")
    print(f"  Error rate:   {np.mean(df['err_rate'][tail_idx:]):.3f} ± {np.std(df['err_rate'][tail_idx:]):.3f}")
    print(f"  Wage parity:  {np.mean(df['wage_parity'][tail_idx:]):.3f} ± {np.std(df['wage_parity'][tail_idx:]):.3f}")
    print(f"  λ (final):    {np.mean(df['lambda_dual'][tail_idx:]):.3f}")

print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)

# Compare error rates
tail_idx = -50
err_A = np.mean(df_A['err_rate'][tail_idx:])
err_B = np.mean(df_B['err_rate'][tail_idx:])
err_C = np.mean(df_C['err_rate'][tail_idx:])

print(f"\n1. Error Rate Control:")
print(f"   - Baseline (A): {err_A:.1%} (constraint active)")
print(f"   - No λ (B):     {err_B:.1%} (no constraint)")
print(f"   - Full (C):     {err_C:.1%} (constraint + novelty)")

# Compare profits
profit_A = np.mean(df_A['profit'][tail_idx:])
profit_B = np.mean(df_B['profit'][tail_idx:])
profit_C = np.mean(df_C['profit'][tail_idx:])

print(f"\n2. Economic Performance:")
print(f"   - Baseline (A): ${profit_A:.2f}/hr")
print(f"   - No λ (B):     ${profit_B:.2f}/hr")
print(f"   - Full (C):     ${profit_C:.2f}/hr")

# Compare wage parity
wp_A = np.mean(df_A['wage_parity'][tail_idx:])
wp_B = np.mean(df_B['wage_parity'][tail_idx:])
wp_C = np.mean(df_C['wage_parity'][tail_idx:])

print(f"\n3. Wage Parity (target = 1.0):")
print(f"   - Baseline (A): {wp_A:.3f}")
print(f"   - No λ (B):     {wp_B:.3f}")
print(f"   - Full (C):     {wp_C:.3f}")

print("\n" + "="*60)
