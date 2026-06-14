import warnings
from time import perf_counter

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
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


def get_estimator(Cs=10, cv=5, n_jobs=-2, use_splines=True):
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
                LogisticRegressionCV(
                    Cs=Cs,
                    l1_ratios=[0.0],
                    cv=cv,
                    solver="newton-cholesky",
                    scoring="neg_log_loss",
                    use_legacy_attributes=False,
                    # refit=False,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


# %%

def run_single_benchmark(data_str, use_splines=True, verbose=1):
    if verbose > 0:
        print(f"running benchmark for {data_str}")
    data_info = DATA_INFOS[data_str]
    meta_data = load_data(data_info=data_info)
    X, y = meta_data["X"], meta_data["y"]

    n_samples, n_features = X.shape
    n_classes = np.unique(y).size

    params = dict(use_splines=use_splines)
    if data_info.n_jobs is not None:
        params["n_jobs"] = data_info.n_jobs
    Cs = np.logspace(-6, 7, 128)
    cv = KFold(n_splits=5, shuffle=True, random_state=4321)
    estimator = get_estimator(Cs=Cs, cv=cv, **params)

    # Trace of Gram matrix; without pipeline transformation it would be
    # trace(X'X) = (X ** 2).sum() = Frobenius norm of X
    X_full = estimator[:-1].fit_transform(X)
    if sp.issparse(X_full):
        X_full = sp.csr_array(X_full)
        trace_gram = sp.linalg.norm(X_full, ord="fro") ** 2
    else:
        trace_gram = np.linalg.norm(X_full, ord="fro") ** 2
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
        estimator.fit(X, y)

    toc = perf_counter()
    duration = toc - tic
    if verbose > 0:
        print(f"  {duration:5.2g} seconds for estimator.fit with solver {estimator[-1].solver}")

    result_list = []
    for i, C in enumerate(estimator[-1].Cs_):
        result_list.append(
            {
                "dataset": meta_data["dataset_name"],
                "data_id": data_info.data_id,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_splits": 5,
                "C": C,
                "log_loss": -np.nanmean(estimator[-1].scores_[:, 0, i]),
                "log_loss_std": np.nanstd(estimator[-1].scores_[:, 0, i]),
                "n_features_full": n_features_full,
                "trace_gram": trace_gram,
                "fit_time": duration,
                "solver": estimator[-1].solver,
            }
        )

    return result_list


# %%

results = []
from pathlib import Path
file = Path("logistic_regression_optimal_penalty.csv")
file.unlink()
for x in list(DATA_INFOS.keys()):
    run = mem.cache(run_single_benchmark)(x, use_splines=False, verbose=1)
    results.append(run)
    df = pl.DataFrame([item for item in run])
    if file.exists():
        df_old = pl.read_csv(file)
        df = pl.concat([df, df_old])  # new on top
    df.write_csv(file)

# %%

df_all = pl.DataFrame([item for run in results for item in run])
df_all.write_parquet("logistic_regression_paths.parquet")
