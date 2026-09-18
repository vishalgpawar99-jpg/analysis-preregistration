# ==============================================================================
# 1. SETUP & DATASET GENERATION
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
# Group A & B: Continuous data (Group A normal, Group B right-skewed)
# Region & Education: Categorical factors for ANOVA
n = 100

data = pd.DataFrame({
    'Group_A': np.random.normal(loc=50, scale=10, size=n),
    'Group_B': np.random.exponential(scale=10, size=n) + 30,
    'Performance_Score': np.random.normal(loc=75, scale=12, size=n),
    'Region': np.random.choice(['North', 'South', 'East'], size=n),
    'Education_Level': np.random.choice(['Bachelors', 'Masters'], size=n)
})

# Introduce factorial differences in Performance_Score for ANOVA demonstration
data.loc[data['Region'] == 'South', 'Performance_Score'] += 8
data.loc[(data['Region'] == 'South') & (data['Education_Level'] == 'Masters'), 'Performance_Score'] += 5

print("Dataset Overview:")
print(data.head())