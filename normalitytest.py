# ==============================================================================
# 2. NORMALITY TESTS (Shapiro-Wilk & Kolmogorov-Smirnov)
# ==============================================================================
"""
Null Hypothesis (H0): The sample distribution is normal.
Alternative Hypothesis (H1): The sample distribution is not normal.
Alpha level = 0.05
"""

# Shapiro-Wilk Test
stat_a_sw, p_a_sw = stats.shapiro(data['Group_A'])
stat_b_sw, p_b_sw = stats.shapiro(data['Group_B'])

# Kolmogorov-Smirnov Test (standardized comparison)
stat_a_ks, p_a_ks = stats.kstest(data['Group_A'], 'norm', args=(data['Group_A'].mean(), data['Group_A'].std()))
stat_b_ks, p_b_ks = stats.kstest(data['Group_B'], 'norm', args=(data['Group_B'].mean(), data['Group_B'].std()))

print("--- Normality Test Results ---")
print(f"Group A - Shapiro-Wilk: W={stat_a_sw:.4f}, p={p_a_sw:.4f} (Normal: {p_a_sw > 0.05})")
print(f"Group B - Shapiro-Wilk: W={stat_b_sw:.4f}, p={p_b_sw:.4f} (Normal: {p_b_sw > 0.05})")
print(f"Group A - Kolmogorov-Smirnov: D={stat_a_ks:.4f}, p={p_a_ks:.4f} (Normal: {p_a_ks > 0.05})")
print(f"Group B - Kolmogorov-Smirnov: D={stat_b_ks:.4f}, p={p_b_ks:.4f} (Normal: {p_b_ks > 0.05})")

# Visualizing Distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data['Group_A'], kde=True, color='skyblue')
plt.title('Group A Distribution (Parametric Target)')

plt.subplot(1, 2, 2)
sns.histplot(data['Group_B'], kde=True, color='salmon')
plt.title('Group B Distribution (Non-Parametric Target)')

plt.tight_layout()
plt.show()