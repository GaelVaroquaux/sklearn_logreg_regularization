import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import joblib

mem = joblib.Memory('joblib_cache')


@dataclass
class DataInfo:
    data_name: str
    data_id: int
    tab_arena: bool = False
    columns_to_remove: list[str] | None = None
    n_jobs: int | None = None

# datasets with the longest fit time:
# 1. ATLAS-Higgs-Boson-Machine-Learning-Challenge-2014 890s
# 2. Higgs-LeoGrin 760s
# 3. sf-police-incidents 280s
# 4. APSFailure 220s
# 5. porto_seguro 200s
# 6. Bioresponse 200s

DATA_INFOS = {
    # START Tab-Arena, see https://arxiv.org/html/2506.16791v4#A2.T2
    # only classification datasets
    "blood-transfusion-service-center": DataInfo(
        data_name="blood-transfusion-service-center", data_id=46913, tab_arena=True
    ),
    "diabetes": DataInfo(data_name="diabetes", data_id=46921, tab_arena=True),  # altenative ID=37
    "anneal": DataInfo(data_name="anneal", data_id=46906, tab_arena=True),
    "credit-g": DataInfo(data_name="credit-g", data_id=46918, tab_arena=True),  # alternative ID=31
    "maternal_health_risk": DataInfo(data_name="maternal_health_risk", data_id=46941, tab_arena=True),
    "qsar-biodeg": DataInfo(data_name="qsar-biodeg", data_id=46952, tab_arena=True),
    "website_phishing": DataInfo(data_name="website_phishing", data_id=46963, tab_arena=True),
    "Fitness_Club": DataInfo(data_name="Fitness_Club", data_id=46927, tab_arena=True),
    "MIC": DataInfo(data_name="MIC", data_id=46980, tab_arena=True),
    "Is-this-a-good-customer": DataInfo(data_name="Is-this-a-good-customer", data_id=46938, tab_arena=True),
    "Marketing_Campaign": DataInfo(data_name="Marketing_Campaign", data_id=46940, tab_arena=True),
    "hazelnut-spread-contaminant-detection": DataInfo(
        data_name="hazelnut-spread-contaminant-detection", data_id=46930, tab_arena=True
    ),
    "seismic-bumps": DataInfo(data_name="seismic-bumps", data_id=46956, tab_arena=True),
    "splice": DataInfo(data_name="splice", data_id=46958, tab_arena=True),
    "Bioresponse": DataInfo(data_name="Bioresponse", data_id=46912, tab_arena=True),
    "hiva_agnostic": DataInfo(data_name="hiva_agnostic", data_id=46933, tab_arena=True),
    "students_dropout_and_academic_success": DataInfo(
        data_name="students_dropout_and_academic_success", data_id=46960, tab_arena=True
    ),
    "churn": DataInfo(data_name="churn", data_id=46915, tab_arena=True),  # altenative ID=40701
    "polish_companies_bankruptcy": DataInfo(
        data_name="polish_companies_bankruptcy", data_id=46950, tab_arena=True
    ),
    "taiwanese_bankruptcy_prediction": DataInfo(
        data_name="taiwanese_bankruptcy_prediction", data_id=46962, tab_arena=True
    ),
    "NATICUSdroid": DataInfo(data_name="NATICUSdroid", data_id=46969, tab_arena=True),
    "coil2000_insurance_policies": DataInfo(
        data_name="coil2000_insurance_policies", data_id=46916, tab_arena=True
    ),
    "Bank_Customer_Churn": DataInfo(data_name="Bank_Customer_Churn", data_id=46911, tab_arena=True),
    "heloc": DataInfo(data_name="heloc", data_id=46911, tab_arena=True),
    "jm1": DataInfo(data_name="jm1", data_id=46979, tab_arena=True),
    "E-CommereShippingData": DataInfo(data_name="E-CommereShippingData", data_id=46924, tab_arena=True),
    "online_shoppers_intention": DataInfo(data_name="online_shoppers_intention", data_id=46947, tab_arena=True),
    "in_vehicle_coupon_recommendation": DataInfo(
        data_name="in_vehicle_coupon_recommendation", data_id=46937, tab_arena=True
    ),
    "HR_Analytics_Job_Change_of_Data_Scientists": DataInfo(
        data_name="HR_Analytics_Job_Change_of_Data_Scientists", data_id=46935, tab_arena=True
    ),
    "credit_card_clients_default": DataInfo(data_name="credit_card_clients_default", data_id=46919, tab_arena=True),
    "Amazon_employee_access": DataInfo(data_name="Amazon_employee_access", data_id=46905, tab_arena=True),
    "bank-marketing": DataInfo(data_name="bank-marketing", data_id=46910, tab_arena=True),
    "kddcup09_appetency": DataInfo(data_name="kddcup09_appetency", data_id=46939, tab_arena=True),  # alternative ID=1111
    "Diabetes130US": DataInfo(data_name="Diabetes130US", data_id=46922, tab_arena=True),
    "APSFailure": DataInfo(data_name="APSFailure", data_id=46908, tab_arena=True),
    "SDSS17": DataInfo(data_name="SDSS17", data_id=46955, tab_arena=True),
    "customer_satisfaction_in_airline": DataInfo(
        data_name="customer_satisfaction_in_airline", data_id=46920, tab_arena=True
    ),
    "GiveMeSomeCredit": DataInfo(data_name="GiveMeSomeCredit", data_id=46929, tab_arena=True),
    # END Tab-Arena
    # Some more useful or known datasets, some taken from https://github.com/thomasjpfan/sk_encoder_cv
    "iris": DataInfo(data_name="iris", data_id=61),  # n_samples=150, n_features=4, n_classes=3
    "dresses_sales": DataInfo(data_name="dresses_sales", data_id=23381),  # n_samples=500, n_features=12, n_classes=2
    "telco": DataInfo(data_name="telco", data_id=42178),  # n_samples=7_043, n_features=19, n_classes=2
    "SpeedDating": DataInfo(data_name="SpeedDating", data_id=40536),  # n_samples=8_378, n_features=120, n_classes=2
    "kdd_internet_usage": DataInfo(data_name="kdd_internet_usage", data_id=981),  # n_samples=10_108, n_features=68, n_classes=2
    "phishing_websites": DataInfo(data_name="phishing_websites", data_id=4534),  # n_samples=11_055, n_features=30, n_classes=2
    "rl": DataInfo(data_name="rl", data_id=41160),  # n_samples=31_406, n_features=22, n_classes=2
    # RESOURCE: An ID for each resource
    "amazon_access": DataInfo(
        data_name="amazon_access", data_id=4135, columns_to_remove=["RESOURCE"]
    ),  # n_sampels=32_769, n_features=9, n_classes=2
    "nomao": DataInfo(data_name="nomao", data_id=1486),  # n_samples=34_465, n_features=117, n_classes=2
    "adult": DataInfo(data_name="adult", data_id=179),  # n_samples=48_842, n_features=14, n_classes=2
    "kicks": DataInfo(data_name="kicks", data_id=41162),  # n_samples=72_983, n_features=32, n_classes=2
    "census_income_kdd": DataInfo(data_name="census_income_kdd", data_id=42750),  # n_samples=199_523, n_features=41
    "porto_seguro": DataInfo(data_name="porto_seguro", data_id=42742),  # n_samples=595_212 n_features=57, n_classes=2
    "ATLAS-Higgs-Boson-Machine-Learning-Challenge-2014": DataInfo(
        data_name="ATLAS-Higgs-Boson-Machine-Learning-Challenge-2014", data_id=45550
    ),  # n_samples=818_238, n_features=30
    "Higgs-LeoGrin": DataInfo(data_name="Higgs-LeoGrin", data_id=44129),  # n_samples=940_160, n_features=24, n_classes=2
    "sf-police-incidents": DataInfo(data_name="sf-police-incidents", data_id=42732),  # n_samples=2_215_023, n_features=8, n_classes=2
    # Higgs causes: python MallocStackLogging: can't turn off malloc stack logging because it was not enabled
    # "Higgs": DataInfo(data_name="Higgs", data_id=45570, n_jobs=1),  # n_samples=11_000_000, n_features=28, n_classes=2
    "electrity": DataInfo(data_name="electrity", data_id=43945),
}


def fetch_openml_and_clean(data_info: DataInfo, verbose=1):
    if verbose > 0:
        print(f"  fetching and loading {data_info.data_name} dataset from openml")
    X, y = fetch_openml(data_id=data_info.data_id, return_X_y=True, as_frame=True)
    X = pl.from_pandas(X, nan_to_null=True)

    if data_info.columns_to_remove:
        X = X.drop(data_info.columns_to_remove)

    return X, y


def load_data(data_info):
    X, y = fetch_openml_and_clean(data_info)
    n_cats = X.select(cs.string(include_categorical=True)).shape[1]
    n_samples, n_features = X.shape

    return {
        "X": X,
        "y": y,
        "dataset_name": data_info.data_name,
        "categorical features": n_cats,
        "n_features": n_features,
        "n_samples": n_samples,
        "openml_url": f"https://www.openml.org/d/{data_info.data_id}",
    }


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
for x in DATA_INFOS.keys():
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
