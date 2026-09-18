# ==============================================================================
# 4. ONE-WAY & TWO-WAY ANOVA WITH TUKEY HSD
# ==============================================================================
"""
One-Way ANOVA:
H0: Means of Performance_Score across all Regions are equal.
H1: At least one Region mean is different.

Two-Way ANOVA:
H0: No main effects or interaction effects between Region and Education_Level.
H1: Significant main effect or interaction effect exists.
"""

# One-Way ANOVA
model_1way = ols('Performance_Score ~ C(Region)', data=data).fit()
anova_1way = sm.stats.anova_lm(model_1way, typ=2)

# Two-Way ANOVA with Interaction
model_2way = ols('Performance_Score ~ C(Region) * C(Education_Level)', data=data).fit()
anova_2way = sm.stats.anova_lm(model_2way, typ=2)

print("--- One-Way ANOVA Table ---")
print(anova_1way)
print("\n--- Two-Way ANOVA Table ---")
print(anova_2way)

# Tukey HSD Post-Hoc Analysis for Region
tukey = pairwise_tukeyhsd(endog=data['Performance_Score'], groups=data['Region'], alpha=0.05)
print("\n--- Tukey HSD Post-Hoc Test ---")
print(tukey)

# Plotting Group Differences for ANOVA
plt.figure(figsize=(10, 5))
sns.boxplot(x='Region', y='Performance_Score', hue='Education_Level', data=data, palette='Set2')
plt.title('Performance Score by Region & Education Level')
plt.xlabel('Region')
plt.ylabel('Performance Score')
plt.show()



