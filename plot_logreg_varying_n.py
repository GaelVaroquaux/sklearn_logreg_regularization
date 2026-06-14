"""
Load the logloss results from the sweeps of C parameters in logistic-regression with varying n, and plot them as a function of different parametrization of the regularization.
"""

# %%
import matplotlib.pyplot as plt

# %%
import pandas as pd

df = pd.read_parquet("logistic_regression_varying_n_paths.parquet")

df

# %%
# plot the first dataset
df_subset = df[df['data_id'] == df['data_id'].unique()[0]]

# The parameterization in C
for train_size in df_subset['n_samples'].unique():
    df_train_size = df_subset[df_subset['n_samples'] == train_size]
    plt.plot(df_train_size['C'], df_train_size['log_loss'], label=f"train_size={train_size}")
plt.xlabel('Regularization parameter C')
plt.ylabel('Logloss')
plt.title('Logloss vs Regularization parameter C for dataset {}'.format(df_subset['dataset'].unique()[0]))
plt.xscale('log')
#plt.yscale('log')
plt.legend()


# The parameterization in alpha
plt.figure()
for train_size in df_subset['n_samples'].unique():
    df_train_size = df_subset[df_subset['n_samples'] == train_size]
    plt.plot(df_train_size['alpha'], df_train_size['log_loss'], label=f"train_size={train_size}")
plt.xlabel('Regularization parameter alpha')
plt.ylabel('Logloss')
plt.title('Logloss vs Regularization parameter alpha for dataset {}'.format(df_subset['dataset'].unique()[0]))
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.show()


# %%
# Now do a pivot table across all datasets to have the logloss as a function of C for each dataset and plot them together.
