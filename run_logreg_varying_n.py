import warnings
from time import perf_counter

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy
import scipy.sparse as sp
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import joblib

# Call git to get the current branch name of scikit-learn (installed in ~/dev/scikit-learn)
import subprocess
import os
sklearn_dir = os.path.expanduser("~/dev/scikit-learn")
sklearn_branch = subprocess.check_output(["git", "-C", sklearn_dir, "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")

mem = joblib.Memory(f'joblib_cache_{sklearn_branch}')

from data_loading import DATA_INFOS, load_data


def get_estimator(use_splines=True, estimator_kw={}):
    cat_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
        sparse_output=True,
    )
    if use_splines:
        num_encoder = make_pipeline(
            SimpleImputer(strategy="constant"),
            SplineTransformer(sparse_output=True),
        )
    else:
        num_encoder = SimpleImputer(strategy="constant")

    def select_cat(X):
        cols = X.select(cs.string(include_categorical=True)).columns
        cols += [
            c for c in X.select(cs.numeric() | cs.temporal()).columns
            if X.get_column(c).head(10_000).n_unique() <= 10
        ]
        return cols

    prep = ColumnTransformer(
        [("cat", cat_encoder, select_cat)],
        remainder=num_encoder,
    )

    return Pipeline(
        [
            ("prep", prep),
            (
                "est",
                LogisticRegression(
                    l1_ratio=0.0,
                    solver="newton-cholesky",
                    **estimator_kw,
                ),
            ),
        ]
    )


# %%

def run_single_benchmark(data_str, param_sweep=("C", [0.01, 0.1, 1, 10, 100]), fraction=1, use_splines=True, verbose=1):
    """
    Parameters
    ----------
    data_str : str
        The name of the dataset to run the benchmark on. Must be a key in DATA_INFOS.
    param_sweep : tuple: (name, list of floats)
        The name of the parameter to sweep over and the list of values to sweep over. For example, ("C", [0.01, 0.1, 1, 10, 100]) would sweep over the regularization parameter C in logistic regression.
    fraction : float, optional
        The fraction of the dataset to use for the benchmark. Must be in (0, 1]. Default is 1 (use the full dataset).
    use_splines : bool, optional
        Whether to use splines for the numerical features. Default is True.
    verbose : int, optional
        The verbosity level. Default is 1 (print progress).
    """
    if verbose > 0:
        print(f"running benchmark for {data_str}, fraction {fraction}")
    data_info = DATA_INFOS[data_str]
    meta_data = load_data(data_info=data_info)
    X, y = meta_data["X"], meta_data["y"]

    # Use only a fraction of the dataset if specified
    if fraction < 1:
        n_samples = X.shape[0]
        n_samples_to_use = int(n_samples * fraction)
        if verbose > 0:
            print(f"Using only {n_samples_to_use} samples out of {n_samples} ({fraction*100:.1f}%)")
        X = X.head(n_samples_to_use)
        y = y.head(n_samples_to_use)

    n_samples, n_features = X.shape
    n_classes = np.unique(y).size




    result_list = []
    param_name, param_values = param_sweep
    for i, param_value in enumerate(param_values):
        params = dict(use_splines=use_splines)
        params['estimator_kw'] = {param_name: param_value}
        if data_info.n_jobs is not None:
            params["n_jobs"] = data_info.n_jobs
        cv = KFold(n_splits=3, shuffle=True, random_state=4321)
        estimator = get_estimator(**params)

        X_full = estimator[:-1].fit_transform(X)
        n_features_full = X_full.shape[1]
        del X_full  # help memory pressure

        if n_features_full > 100:
            estimator[-1].solver = "newton-cg"
        if verbose > 0:
            print(f"  {n_samples=} {n_features=} {n_classes=} n_features_full={n_features_full}")

        tic = perf_counter()
        print('Starting fit')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            scores = cross_val_score(estimator, X, y, cv=cv, scoring="neg_log_loss",
                                     n_jobs=data_info.n_jobs, verbose=2)

        toc = perf_counter()
        duration = toc - tic

        result_list.append(
            {
                "dataset": meta_data["dataset_name"],
                "data_id": data_info.data_id,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_splits": 5,
                param_name: param_value,
                "log_loss": -np.nanmean(scores),
                "log_loss_std": np.nanstd(scores),
                "n_features_full": n_features_full,
                "fit_time": duration,
                "solver": estimator[-1].solver,
            }
        )
    
    if verbose > 0:
        print(f"  {duration:5.2g} seconds for estimator.fit with solver {estimator[-1].solver}")

    return result_list


DATASETS = ["porto_seguro", ]# "jannis_large", "census_income_kdd"]

FRACTIONS = [.1, .25, .5, .75,  1]

# %%

C_sweep = ("C", np.logspace(-4, 2, 16))
alpha_sweep = ("alpha", np.logspace(-5, 1, 16))

from tqdm import tqdm

results = []
with tqdm(total=len(DATASETS) * len(FRACTIONS), desc="Overall", position=1, leave=True) as pbar_outer:
    for x in tqdm(DATASETS, desc="Datasets", position=0, leave=True):
        for fraction in tqdm(FRACTIONS, desc="Fractions", position=1, leave=False):
            run = mem.cache(run_single_benchmark)(x, param_sweep=C_sweep, fraction=fraction, use_splines=False, verbose=1)
            results.append(run)
            run = mem.cache(run_single_benchmark)(x, param_sweep=alpha_sweep, fraction=fraction, use_splines=False, verbose=1)
            results.append(run)

            pbar_outer.update(1)

# %%

df_all = pl.DataFrame([item for run in results for item in run])
df_all.write_parquet("logistic_regression_varying_n_paths.parquet")
