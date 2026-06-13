import warnings
from time import perf_counter
from urllib.error import URLError

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import joblib

mem = joblib.Memory('joblib_cache')

from data_loading import DATA_INFOS, load_data


def get_logreg(use_splines=True):
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
                LogisticRegression(),
            ),
        ]
    )



# %%

def run_single_logreg(data_str, use_splines=True, verbose=1):
    if verbose > 0:
        print(f"running benchmark for {data_str}")
    data_info = DATA_INFOS[data_str]
    meta_data = load_data(data_info=data_info)
    X, y = meta_data["X"], meta_data["y"]

    n_samples, n_features = X.shape
    n_classes = np.unique(y).size

    params = dict(use_splines=use_splines)
    estimator = get_logreg(**params)

    print('Starting fit')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
        results = cross_validate(estimator, X, y, cv=5, n_jobs=-1,
                                 scoring=['neg_log_loss'])

    duration = np.mean(results['fit_time'])

    if verbose > 0:
        print(f"  {duration:5.2g} seconds for estimator.fit with solver {estimator[-1].solver}")

    test_scores = results['test_neg_log_loss']

    output = {
                "dataset": meta_data["dataset_name"],
                "data_id": data_info.data_id,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_splits": len(test_scores),
                "log_loss": -np.nanmean(test_scores),
                "log_loss_std": np.nanstd(test_scores),
                "fit_time": duration,
                "solver": estimator[-1].solver,
            }

    return output


# %%

results = []
from pathlib import Path
file = Path("logistic_regression_default.csv")
if file.exists():
    file.unlink()

for x in list(DATA_INFOS.keys())[::-1]:
    try:
        run = mem.cache(run_single_logreg)(x, use_splines=False, verbose=1)
    except URLError:
        # openML troubles
        print(f'******** skipping {x} URLError/HTTPError ************')
        continue
    results.append(run)
    df = pl.DataFrame([item for item in run])
    if file.exists():
        df_old = pl.read_csv(file)
        df = pl.concat([df, df_old])  # new on top
    df.write_csv(file)

# %%

df_all = pl.DataFrame(results)
df_all.write_parquet("logistic_regression_default.parquet")
