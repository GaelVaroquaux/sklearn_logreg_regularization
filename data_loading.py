"""
A list of datasets and corresponding loading utility
"""

from dataclasses import dataclass

import polars as pl
import polars.selectors as cs
from sklearn.datasets import fetch_openml
from urllib.error import URLError


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
    #"heloc2": DataInfo(data_name="heloc", data_id=46932, tab_arena=True),
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
    # START Grinsztajn et al. - "Why tree-based models still outperform deep learning on typical tabular data"
    # https://arxiv.org/abs/2207.08815 - OpenML suites 334 (cat+num) and 337 (num-only) classification
    "credit": DataInfo(data_name="credit", data_id=44089),  # n_samples=16_714, n_features=11, n_classes=2
    "covertype_binary": DataInfo(data_name="covertype", data_id=44121),  # n_samples=566_602, n_features=11, n_classes=2
    "pol": DataInfo(data_name="pol", data_id=44122),  # n_samples=10_082, n_features=27, n_classes=2
    "house_16H": DataInfo(data_name="house_16H", data_id=44123),  # n_samples=13_488, n_features=17, n_classes=2
    "MagicTelescope": DataInfo(data_name="MagicTelescope", data_id=44125),  # n_samples=13_376, n_features=11, n_classes=2
    "MiniBooNE": DataInfo(data_name="MiniBooNE", data_id=44128),  # n_samples=72_998, n_features=51, n_classes=2
    "eye_movements": DataInfo(data_name="eye_movements", data_id=44130),  # n_samples=7_608, n_features=21, n_classes=2
    "default-of-credit-card-clients": DataInfo(data_name="default-of-credit-card-clients", data_id=45020),  # n_samples=13_272, n_features=21, n_classes=2
    "jannis": DataInfo(data_name="jannis", data_id=45021),  # n_samples=57_580, n_features=55, n_classes=2
    "california": DataInfo(data_name="california", data_id=45028),  # n_samples=20_634, n_features=9, n_classes=2 (binary: above/below median price)
    "albert": DataInfo(data_name="albert", data_id=45035),  # n_samples=58_252, n_features=32, n_classes=2
    "road-safety": DataInfo(data_name="road-safety", data_id=45038),  # n_samples=111_762, n_features=33, n_classes=2
    "compas-two-years": DataInfo(data_name="compas-two-years", data_id=45039),  # n_samples=4_966, n_features=12, n_classes=2
    # END Grinsztajn et al.
    # START Holzmuller et al. - "Better by default: Strong pre-tuned MLPs and boosted trees on tabular data"
    # https://arxiv.org/abs/2407.04491 - AutoML Benchmark OpenML suite 271 (classification)
    "vehicle": DataInfo(data_name="vehicle", data_id=54),  # n_samples=846, n_features=19, n_classes=4
    "mozilla4": DataInfo(data_name="mozilla4", data_id=1046),  # n_samples=15_545, n_features=6, n_classes=2
    "pc4": DataInfo(data_name="pc4", data_id=1049),  # n_samples=1_458, n_features=38, n_classes=2
    "kc1": DataInfo(data_name="kc1", data_id=1067),  # n_samples=2_109, n_features=22, n_classes=2
    "airlines": DataInfo(data_name="airlines", data_id=1169),  # n_samples=539_383, n_features=8, n_classes=2
    "phoneme": DataInfo(data_name="phoneme", data_id=1489),  # n_samples=5_404, n_features=6, n_classes=2
    "numerai28.6": DataInfo(data_name="numerai28.6", data_id=23517),  # n_samples=96_320, n_features=22, n_classes=2
    "connect-4": DataInfo(data_name="connect-4", data_id=40668),  # n_samples=67_557, n_features=43, n_classes=3
    "dna": DataInfo(data_name="dna", data_id=40670),  # n_samples=3_186, n_features=181, n_classes=3
    # Skipping because it fails
    #"shuttle": DataInfo(data_name="shuttle", data_id=40685),  # n_samples=58_000, n_features=10, n_classes=7
    "Satellite": DataInfo(data_name="Satellite", data_id=40900),  # n_samples=5_100, n_features=37, n_classes=6
    "car": DataInfo(data_name="car", data_id=40975),  # n_samples=1_728, n_features=7, n_classes=4
    "Australian": DataInfo(data_name="Australian", data_id=40981),  # n_samples=690, n_features=15, n_classes=2
    "steel-plates-fault": DataInfo(data_name="steel-plates-fault", data_id=40982),  # n_samples=1_941, n_features=28, n_classes=7
    "segment": DataInfo(data_name="segment", data_id=40984),  # n_samples=2_310, n_features=20, n_classes=7
    "jungle_chess_2pcs_raw_endgame_complete": DataInfo(data_name="jungle_chess_2pcs_raw_endgame_complete", data_id=41027),  # n_samples=44_819, n_features=7, n_classes=3
    "christine": DataInfo(data_name="christine", data_id=41142),  # n_samples=5_418, n_features=1_637, n_classes=2
    "jasmine": DataInfo(data_name="jasmine", data_id=41143),  # n_samples=2_984, n_features=145, n_classes=2
    "madeline": DataInfo(data_name="madeline", data_id=41144),  # n_samples=3_140, n_features=260, n_classes=2
    "philippine": DataInfo(data_name="philippine", data_id=41145),  # n_samples=5_832, n_features=309, n_classes=2
    "sylvine": DataInfo(data_name="sylvine", data_id=41146),  # n_samples=5_124, n_features=21, n_classes=2
    "albert_large": DataInfo(data_name="albert", data_id=41147),  # n_samples=425_240, n_features=79, n_classes=2; larger/rawer version than id=45035
    "MiniBooNE_large": DataInfo(data_name="MiniBooNE", data_id=41150),  # n_samples=130_064, n_features=51, n_classes=2; larger/rawer version than id=44128
    "ada": DataInfo(data_name="ada", data_id=41156),  # n_samples=4_147, n_features=49, n_classes=2
    "gina": DataInfo(data_name="gina", data_id=41158),  # n_samples=3_153, n_features=971, n_classes=2
    "guillermo": DataInfo(data_name="guillermo", data_id=41159),  # n_samples=20_000, n_features=4_297, n_classes=2
    "riccardo": DataInfo(data_name="riccardo", data_id=41161),  # n_samples=20_000, n_features=4_297, n_classes=2
    "dilbert": DataInfo(data_name="dilbert", data_id=41163),  # n_samples=10_000, n_features=2_001, n_classes=5
    "fabert": DataInfo(data_name="fabert", data_id=41164),  # n_samples=8_237, n_features=801, n_classes=7
    # Download currently failing
    #"robert": DataInfo(data_name="robert", data_id=41165),  # n_samples=10_000, n_features=7_201, n_classes=10
    "volkert": DataInfo(data_name="volkert", data_id=41166),  # n_samples=58_310, n_features=181, n_classes=10
    # Terribly slow to converge with LogisticRegressionCV
    #"dionis": DataInfo(data_name="dionis", data_id=41167),  # n_samples=416_188, n_features=61, n_classes=355
    "jannis_large": DataInfo(data_name="jannis", data_id=41168),  # n_samples=83_733, n_features=55, n_classes=4; larger/rawer version than id=45021
    "helena": DataInfo(data_name="helena", data_id=41169),  # n_samples=65_196, n_features=28, n_classes=100
    # The below fails to download
    # "Click_prediction_small": DataInfo(data_name="Click_prediction_small", data_id=42733),  # n_samples=39_948, n_features=12, n_classes=2
    # END Holzmuller et al.
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
    try:
        X, y = fetch_openml(data_id=data_info.data_id, return_X_y=True, as_frame=True)
    except ValueError as ex:
        print(ex)
        # SuperClass of HTTPError
        raise URLError(reason=f"fetch_openml failed on {data_info.data_name}")
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




