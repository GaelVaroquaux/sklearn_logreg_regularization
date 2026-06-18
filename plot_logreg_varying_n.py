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

# To min-max scale the logloss values to put them on a similar scale across datasets, we can use the following function.
def normalize_log_loss(log_loss):
    log_loss = log_loss - log_loss.min()
    log_loss = log_loss / log_loss.max()
    return log_loss

# The parameterization in C
for train_size in df_subset['n_samples'].unique():
    df_train_size = df_subset[df_subset['n_samples'] == train_size]
    plt.plot(df_train_size['C'], normalize_log_loss(df_train_size['log_loss']), label=f"train_size={train_size}")
plt.xlabel('Regularization parameter C')
plt.ylabel('Logloss (normalized)')
plt.title('Logloss vs Regularization parameter C for dataset {}'.format(df_subset['dataset'].unique()[0]))
plt.xscale('log')
plt.legend()


# The parameterization in alpha
plt.figure()
for train_size in df_subset['n_samples'].unique():
    df_train_size = df_subset[df_subset['n_samples'] == train_size]
    plt.plot(df_train_size['alpha'], normalize_log_loss(df_train_size['log_loss']), label=f"train_size={train_size}")
plt.xlabel('Regularization parameter alpha')
plt.ylabel('Logloss (normalized)')
plt.title('Logloss vs Regularization parameter alpha for dataset {}'.format(df_subset['dataset'].unique()[0]))
plt.xscale('log')
plt.legend()
plt.show()

# %%

# plot the second dataset
df_subset = df[df['data_id'] == df['data_id'].unique()[1]]

# To min-max scale the logloss values to put them on a similar scale across datasets, we can use the following function.
def normalize_log_loss(log_loss):
    log_loss = log_loss - log_loss.min()
    log_loss = log_loss / log_loss.max()
    return log_loss

# The parameterization in C
for train_size in df_subset['n_samples'].unique():
    df_train_size = df_subset[df_subset['n_samples'] == train_size]
    plt.plot(df_train_size['C'], normalize_log_loss(df_train_size['log_loss']), label=f"train_size={train_size}")
plt.xlabel('Regularization parameter C')
plt.ylabel('Logloss (normalized)')
plt.title('Logloss vs Regularization parameter C for dataset {}'.format(df_subset['dataset'].unique()[0]))
plt.xscale('log')
plt.legend()


# The parameterization in alpha
plt.figure()
for train_size in df_subset['n_samples'].unique():
    df_train_size = df_subset[df_subset['n_samples'] == train_size]
    plt.plot(df_train_size['alpha'], normalize_log_loss(df_train_size['log_loss']), label=f"train_size={train_size}")
plt.xlabel('Regularization parameter alpha')
plt.ylabel('Logloss (normalized)')
plt.title('Logloss vs Regularization parameter alpha for dataset {}'.format(df_subset['dataset'].unique()[0]))
plt.xscale('log')
plt.legend()
plt.show()


# %%
