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

# The results for the different branches are stored in different parquet files. We load them and concatenate them into a single dataframe.
import glob
df_list = []
for file in sorted(glob.glob("logistic_regression_*.parquet")):
    variants = ".".join("_".join(file.split("_")[2:]).split(".")[:-1])
    if variants == "paths":
        continue # hack: bad name convention and thus collision
    this_df = pd.read_parquet(file)
    this_df['variant'] = variants
    df_list.append(this_df)
df = pd.concat(df_list, ignore_index=True)

df

# %%
# Striplot and boxen of the logloss

x_name = "variant"
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.stripplot(x=x_name, y="log_loss", data=df, jitter=True, color="blue", alpha=0.5)
sns.boxenplot(x=x_name, y="log_loss", data=df, color="lightgray")
# Write the percentiles on the boxen plot
for i, variant in enumerate(df[x_name].unique()):
    log_loss_values = df[df[x_name] == variant]['log_loss']
    percentiles = np.percentile(log_loss_values, [25, 50, 75])
    for j, percentile in enumerate(percentiles):
        plt.text(i + .21, percentile, f"{percentile:.2f}", ha='left', va='bottom', fontsize=10, color='black')
plt.title("Cross-validated log-loss")
plt.yscale("log")
plt.xlabel("Choice of regularization parameter")
plt.savefig("loss_with_default_parameters.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# Striplot and boxen of the fit_time

x_name = "variant"
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.stripplot(x=x_name, y="fit_time", data=df, jitter=True, color="blue", alpha=0.5)
sns.boxenplot(x=x_name, y="fit_time", data=df, color="lightgray")
# Write the percentiles on the boxen plot
for i, variant in enumerate(df[x_name].unique()):
    fit_time_values = df[df[x_name] == variant]['fit_time']
    percentiles = np.percentile(fit_time_values, [25, 50, 75])
    for j, percentile in enumerate(percentiles):
        plt.text(i + .21, percentile, f"{percentile:.2f}", ha='left', va='bottom', fontsize=10, color='black')
plt.title("Fit time")
plt.yscale("log")
plt.xlabel("Choice of regularization parameter")
plt.savefig("fit_time_with_default_parameters.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# For the LogisticRegressionCV now

dfcv_list = []
for file in sorted(glob.glob("logistic_regressioncv_default_*.parquet")):
    variants = ".".join("_".join(file.split("_")[3:]).split(".")[:-1])
    if variants == "paths":
        continue # hack: bad name convention and thus collision
    this_df = pd.read_parquet(file)
    this_df['variant'] = variants
    dfcv_list.append(this_df)
dfcv = pd.concat(dfcv_list, ignore_index=True)

dfcv

# %%
# Striplot and boxen of the logloss

x_name = "variant"
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.stripplot(x=x_name, y="log_loss", data=dfcv, jitter=True, color="blue", alpha=0.5)
sns.boxenplot(x=x_name, y="log_loss", data=dfcv, color="lightgray")
# Write the percentiles on the boxen plot
for i, variant in enumerate(dfcv[x_name].unique()):
    log_loss_values = dfcv[dfcv[x_name] == variant]['log_loss']
    percentiles = np.percentile(log_loss_values, [25, 50, 75])
    for j, percentile in enumerate(percentiles):
        plt.text(i + .21, percentile, f"{percentile:.2f}", ha='left', va='bottom', fontsize=10, color='black')
plt.title("Cross-validated log-loss")
plt.yscale("log")
plt.xlabel("Choice of regularization parameter")
plt.savefig("loss_with_default_parameters.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# Striplot and boxen of the fit_time

x_name = "variant"
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.stripplot(x=x_name, y="fit_time", data=dfcv, jitter=True, color="blue", alpha=0.5)
sns.boxenplot(x=x_name, y="fit_time", data=dfcv, color="lightgray")
# Write the percentiles on the boxen plot
for i, variant in enumerate(dfcv[x_name].unique()):
    fit_time_values = dfcv[dfcv[x_name] == variant]['fit_time']
    percentiles = np.percentile(fit_time_values, [25, 50, 75])
    for j, percentile in enumerate(percentiles):
        plt.text(i + .21, percentile, f"{percentile:.2f}", ha='left', va='bottom', fontsize=10, color='black')
plt.title("Fit time")
plt.yscale("log")
plt.xlabel("Choice of regularization parameter")
plt.savefig("fit_time_with_default_parameters.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
