"""
Load the logloss and fit time results from the sweeps of C parameters in
logistic-regression and plot them as a function of different
parametrization of the regularization.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import pandas as pd

df = pd.read_parquet('logistic_regression_default.parquet')

df

# %%
# Striplot and boxen of the logloss

x_name = "solver"
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.stripplot(x=x_name, y="log_loss", data=df, jitter=True, color="blue", alpha=0.5)
sns.boxenplot(x=x_name, y="log_loss", data=df, color="lightgray")
# Write the percentiles on the boxen plot
for i, solver in enumerate(df[x_name].unique()):
    log_loss_values = df[df[x_name] == solver]['log_loss']
    percentiles = np.percentile(log_loss_values, [25, 50, 75])
    for j, percentile in enumerate(percentiles):
        plt.text(i + .21, percentile, f"{percentile:.2f}", ha='left', va='bottom', fontsize=10, color='black')
plt.title("Cross-validated log-loss with default parameters")
plt.yscale("log")
plt.savefig("loss_with_default_parameters.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# Striplot and boxen of the fit_time

x_name = "solver"
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.stripplot(x=x_name, y="fit_time", data=df, jitter=True, color="blue", alpha=0.5)
sns.boxenplot(x=x_name, y="fit_time", data=df, color="lightgray")
# Write the percentiles on the boxen plot
for i, solver in enumerate(df[x_name].unique()):
    fit_time_values = df[df[x_name] == solver]['fit_time']
    percentiles = np.percentile(fit_time_values, [25, 50, 75])
    for j, percentile in enumerate(percentiles):
        plt.text(i + .21, percentile, f"{percentile:.2f}", ha='left', va='bottom', fontsize=10, color='black')
plt.title("Fit time with default parameters")
plt.yscale("log")
plt.savefig("fit_time_with_default_parameters.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
