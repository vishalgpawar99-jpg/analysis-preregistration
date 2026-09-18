# ==============================================================================
# 3. HYPOTHESIS TESTING (t-test & Mann-Whitney U)
# ==============================================================================
"""
Two-Sample T-Test:
H0: mean(Group_A) = mean(Group_B)
H1: mean(Group_A) != mean(Group_B)

Mann-Whitney U Test:
H0: distribution(Group_A) = distribution(Group_B)
H1: distribution(Group_A) != distribution(Group_B)
"""

# Two-Sample Independent T-Test (Parametric)
t_stat, p_ttest = stats.ttest_ind(data['Group_A'], data['Group_B'])

# Mann-Whitney U Test (Non-Parametric)
u_stat, p_mwu = stats.mannwhitneyu(data['Group_A'], data['Group_B'])

# 95% Confidence Interval for mean difference (Group A - Group B)
diff_mean = data['Group_A'].mean() - data['Group_B'].mean()
se_diff = np.sqrt(data['Group_A'].var()/n + data['Group_B'].var()/n)
ci_lower, ci_upper = stats.t.interval(0.95, df=2*n-2, loc=diff_mean, scale=se_diff)

print("--- Two-Sample Test Results ---")
print(f"Two-Sample T-Test: t={t_stat:.4f}, p={p_ttest:.4e}")
print(f"Mann-Whitney U Test: U={u_stat:.4f}, p={p_mwu:.4e}")
print(f"95% Confidence Interval for Mean Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")