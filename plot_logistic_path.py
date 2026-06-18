"""
Load the logloss results from the sweeps of C parameters in
logistic-regression and plot them as a function of different
parametrization of the regularization.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import pandas as pd

df = pd.read_parquet('logistic_regression_paths.parquet')

df

# %%
# A scaling function to put the logloss values on a similar scale across datasets.
def normalize_log_loss(log_loss):
    # Min-max scaling to put the logloss values on a similar scale across datasets.
    log_loss = log_loss - log_loss.min()
    log_loss = log_loss / log_loss.max()
    # Add a small constant to avoid log(0) issues when plotting on a log scale.
    return log_loss + 2e-4


# %%
# Select the data for the first dataset (data_id == 46913) and plot the logloss as a function of the regularization parameter C.

df_subset = df[df['data_id'] == 46913]

plt.plot(df_subset['C'], normalize_log_loss(df_subset['log_loss']))
plt.xlabel('Regularization parameter C')
plt.ylabel('Logloss')
plt.title('Logloss vs Regularization parameter C for dataset 46913')
# Set the x-axis to log scale
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
# Apply the scaling function to the logloss values for each dataset separately.
df['log_loss_normalized'] = df.groupby('dataset')['log_loss'].transform(normalize_log_loss)

# %%
# Now do a pivot table across all datasets to have the logloss as a function of C for each dataset and plot them together.
pivot_table = df.pivot_table(index='dataset', columns='C', values='log_loss_normalized')
# %%
# First a "simple" plot without regriding the data to see how it looks.

# Plot the trim mean and percentiles of the logloss across all datasets as a function of C.
from scipy.stats import trim_mean

trim_mean_log_loss = pivot_table.apply(lambda x: trim_mean(x, proportiontocut=0.1), axis=0)
percentiles_log_loss = pivot_table.quantile([0.25, 0.75])
plt.plot(trim_mean_log_loss.index, trim_mean_log_loss.values, label='Trim Mean Logloss')
plt.fill_between(percentiles_log_loss.columns, percentiles_log_loss.loc[0.25], percentiles_log_loss.loc[0.75], alpha=0.2, label='25th-75th Percentile')
# Plot each dataset with a lightly transparent line
for dataset in pivot_table.index:
    plt.plot(pivot_table.columns, pivot_table.loc[dataset], alpha=0.2, color='gray')
plt.xlabel('Regularization parameter C')
plt.xscale('log')
plt.yscale('log')
plt.ylim(2e-4, 1e0)
plt.ylabel('Logloss')

# %%
def plot_regrided_logloss_vs_scaled_C(
    pivot_table,
    scaling_factor_per_dataset,
    xlabel,
    x_values=None,
    x_min_extrapolation=None,
    x_max_extrapolation=None,
):
    C_values = pivot_table.columns
    regrided_data = []
    if x_min_extrapolation is None:
        x_min_extrapolation = C_values.min() * scaling_factor_per_dataset.min() * 0.2
    if x_max_extrapolation is None:
        x_max_extrapolation = C_values.max() * scaling_factor_per_dataset.max() * 5
    if x_values is None:
        x_values = np.logspace(np.log10(x_min_extrapolation), np.log10(x_max_extrapolation), 100)

    # --
    # First plot the loss curves for each dataset with the scaled C values to visualize the spread of the data.
    # A first axes of a figure to plot the loss curves for each dataset with the scaled C values to visualize the spread of the data.
    plt.figure(figsize=(8, 4))
    # Split the figure into two subplots, taking 3/4 of the width for the first subplot and 1/4 for the second subplot.
    gs = plt.GridSpec(1, 4)
    ax1 = plt.subplot(gs[0, :3])
    ax2 = plt.subplot(gs[0, 3])

    # Plot each dataset with a lightly transparent line
    for dataset in pivot_table.index:
        scaling_factor = scaling_factor_per_dataset[dataset]
        scaled_x = C_values * scaling_factor
        y = pivot_table.loc[dataset].values

        ax1.plot(scaled_x, y, alpha=0.2, color='gray')

        # Add boundary points to help interpolation outside observed range.
        extrapolated_log_loss = np.r_[1, y, 1]
        extrapolated_x_values = np.r_[x_min_extrapolation, scaled_x, x_max_extrapolation]
        regrided_log_loss = np.interp(x_values, extrapolated_x_values, extrapolated_log_loss)
        regrided_data.append(regrided_log_loss)

    regrided_data = np.array(regrided_data)
    mean_log_loss_regrided = trim_mean(regrided_data, axis=0, proportiontocut=0.1)
    percentiles_log_loss_regrided = np.percentile(regrided_data, [25, 75], axis=0)

    ax1.plot(x_values, mean_log_loss_regrided, label='Trim Mean Logloss')
    ax1.fill_between(
        x_values,
        percentiles_log_loss_regrided[0],
        percentiles_log_loss_regrided[1],
        alpha=0.2,
        label='25th-75th Percentile',
    )

    ax1.set_xlabel(xlabel)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(2e-4, 1e0)
    ax1.set_ylabel('Logloss')

    # Now plot the distribution of loss values at a few selected x values to visualize the spread of the data
    argmin_indices = np.argmin(mean_log_loss_regrided)
    min_x_value = x_values[argmin_indices]
    log_loss_at_min_x = regrided_data[:, argmin_indices]
    # Over the 3 consecutive x values around the minimum to similute a small grid-search.
    log_loss_around_min_x = regrided_data[:, max(0, argmin_indices - 4):min(len(x_values), argmin_indices + 5)].min(axis=1)
    # Concatenate these two arrays in a "tall" format dataframe.
    distribution_df = pd.DataFrame({
        'log_loss': np.concatenate([np.log(log_loss_at_min_x), np.log(log_loss_around_min_x)]),
        'x_value': ['at min\n({:.2e})'.format(min_x_value)] * len(log_loss_at_min_x) + ['around\nmin'] * len(log_loss_around_min_x),
    })

    # Overlay a stripplot and a boxen plot to show the distribution of logloss values at the selected x value.
    sns.boxenplot(x='x_value', y='log_loss', data=distribution_df, color='lightgray', ax=ax2)
    sns.stripplot(x='x_value', y='log_loss', data=distribution_df, ax=ax2)
    ax2.set_xlabel('')
    plt.savefig(f'logloss_vs_{xlabel.replace(" ", "_").replace("/", "_divided_by_")}.png', dpi=300, bbox_inches='tight')
    return distribution_df


# %%
# Plots as different parametrizations of the regularization parameter C to see if it aligns better across datasets.

# Accumulate the distributions for a final plot
loss_at_constant_parametrization = {}

# Now plot the same but with the x-axis as C to see if it aligns better across datasets.
constant_one_per_dataset = df.groupby('dataset')['n_samples'].first().copy()
constant_one_per_dataset[:] = 1
name = 'Regularization parameter C'
distributions = plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=constant_one_per_dataset,
    xlabel=name,
)
loss_at_constant_parametrization[name] = distributions

# Now plot the same but with the x-axis as n_samples * C to see if it aligns better across datasets.
n_samples_per_dataset = df.groupby('dataset')['n_samples'].first()
x_values = np.logspace(-2, 7, 100)
name = 'n_samples * Regularization parameter C'
distribution = plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=n_samples_per_dataset,
    xlabel=name,
)
loss_at_constant_parametrization[name] = distribution

# Now plot the same but with the x-axis as C / n_samples to see if it aligns better across datasets.
n_samples_per_dataset = df.groupby('dataset')['n_samples'].first()
x_values = np.logspace(-2, 7, 100)
plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=1. / n_samples_per_dataset,
    xlabel=' Regularization parameter C / n_samples',
)

# Now plot the same but with the x-axis as C / sqrt(n_samples) to see if it aligns better across datasets.
n_samples_per_dataset = df.groupby('dataset')['n_samples'].first()
x_values = np.logspace(-2, 7, 100)
name = ' Regularization parameter C / sqrt n_samples'
distribution = plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=1. / np.sqrt(n_samples_per_dataset),
    xlabel=name,
)
loss_at_constant_parametrization[name] = distribution


# Now plot the same but with the x-axis as C / log(n_samples) to see if it aligns better across datasets.
n_samples_per_dataset = df.groupby('dataset')['n_samples'].first()
x_values = np.logspace(-2, 7, 100)
name = ' Regularization parameter C / log n_samples'
distribution = plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=1. / np.log(n_samples_per_dataset),
    xlabel=name,
)
loss_at_constant_parametrization[name] = distribution

# Now plot the same but with the x-axis as C / trace_gram to see if it aligns better across datasets.
trace_gram_per_dataset = df.groupby('dataset')['trace_gram'].first()
n_features_per_dataset = df.groupby('dataset')['n_features'].first()

# Actually, we want the mean eigenvalue, which is trace_gram / n_features.
mean_eigenvalue_per_dataset = trace_gram_per_dataset / n_features_per_dataset
x_values = np.logspace(-11, 4, 100)
name = 'Regularization parameter C / mean eigenvalue'
distribution = plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=1 / mean_eigenvalue_per_dataset,
    xlabel=name,
)
loss_at_constant_parametrization[name] = distribution


name = 'Regularization parameter C / trace_gram'
distribution = plot_regrided_logloss_vs_scaled_C(
    pivot_table=pivot_table,
    scaling_factor_per_dataset=1 / trace_gram_per_dataset,
    xlabel=name,
)
loss_at_constant_parametrization[name] = distribution

# %%
# Concatenate the dfs in dictionnary loss_at_constant_parametrization into a big df 
big_df = pd.concat(loss_at_constant_parametrization.values(), keys=loss_at_constant_parametrization.keys())
big_df = big_df.reset_index()
# rename the columns to have "name" instead of "level_0"
big_df = big_df.rename(columns={'level_0': 'name'})

# Select only the results not "around min"
results_at_default = big_df[big_df["x_value"] != 'around\nmin']

# Compute the median log_loss for each name to order the boxplots by median log_loss
median_log_loss = results_at_default.groupby("name")["log_loss"].median().sort_values(ascending=False)
results_at_default["name"] = pd.Categorical(results_at_default["name"],
                                            categories=median_log_loss.index,
                                            ordered=True)
# Do stripplots and boxenplots on the results_at_default

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 8))
sns.stripplot(y="name", x="log_loss", data=results_at_default, dodge=True, jitter=True, orient="h", order=median_log_loss.index)
sns.boxenplot(y="name", x="log_loss", data=results_at_default, dodge=True, linewidth=1, orient="h", order=median_log_loss.index)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('comparison_of_parametrizations.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
