# plot_training.py
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# Create plots directory
os.makedirs("plots", exist_ok=True)

xs, lam, err, profit, reward, wpar = [], [], [], [], [], []

with open("logs/dishwashing_run.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(int(row["episode"]))
        lam.append(float(row["lambda_dual"]))
        err.append(float(row["err_rate"]))
        profit.append(float(row["profit"]))
        reward.append(float(row["reward"]))
        wpar.append(float(row["wage_parity"]))

# 1) Lambda trajectory
plt.figure(figsize=(10, 6))
plt.plot(xs, lam, linewidth=1.5)
plt.title("Lagrange Multiplier (λ) over Training", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("λ", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/lambda_trajectory.png", dpi=150)
print("✓ Saved: plots/lambda_trajectory.png")

# 2) Error rate vs target
ERR_TARGET = 0.06
plt.figure(figsize=(10, 6))
plt.plot(xs, err, label="Error rate", linewidth=1.5)
plt.axhline(ERR_TARGET, linestyle="--", color='red', label="Target (e*=6%)", linewidth=2)
plt.title("Error Rate Convergence", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Error rate", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/error_rate.png", dpi=150)
print("✓ Saved: plots/error_rate.png")

# 3) Profit and reward
plt.figure(figsize=(10, 6))
plt.plot(xs, profit, label="Profit ($/hr)", linewidth=1.5, alpha=0.8)
plt.plot(xs, reward, label="Reward", linewidth=1.5, alpha=0.8)
plt.title("Profit & Reward", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("$/hr", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/profit_reward.png", dpi=150)
print("✓ Saved: plots/profit_reward.png")

# 4) Wage parity
plt.figure(figsize=(10, 6))
plt.plot(xs, wpar, linewidth=1.5)
plt.axhline(1.0, linestyle="--", color='green', label="Parity (ŵᵣ/wₕ = 1.0)", linewidth=2)
plt.title("Wage Parity (ŵᵣ / wₕ)", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Parity", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/wage_parity.png", dpi=150)
print("✓ Saved: plots/wage_parity.png")

print("\n✅ All plots saved to plots/ directory")
