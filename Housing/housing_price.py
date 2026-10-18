import sys
assert sys.version_info >= (3, 7)  # Use Python 3.7+
from packaging import version
import sklearn
# UPGRADED VERSION CHECK: You must have 1.2.0+ for sparse_output=False to work!
assert version.parse(sklearn.__version__) >= version.parse("1.2.0"), "Please update Scikit-Learn to 1.2+ (pip install -U scikit-learn)"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tarfile
import urllib.request
import joblib

# -------------------------------------------------------------------
# 1. FETCH AND LOAD DATA
# -------------------------------------------------------------------
print("1. Fetching housing data...")
def housing_price():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing.csv")) 

housing = housing_price()

IMAGES_PATH = Path()/"images"/"end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# -------------------------------------------------------------------
# 2. SPLIT DATA
# -------------------------------------------------------------------
print("2. Splitting data into train and test sets...")
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

start_train_set, start_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (start_train_set, start_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = start_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()

# -------------------------------------------------------------------
# 3. BUILD CUSTOM TRANSFORMERS
# -------------------------------------------------------------------
print("3. Building preprocessing pipelines...")
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class Clustersimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Setting n_init explicitly suppresses warnings in newer sklearn versions
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_name_in):
    return ["ratio"] 

# -------------------------------------------------------------------
# 4. ASSEMBLE PIPELINE
# -------------------------------------------------------------------
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

def ration_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"), 
        FunctionTransformer(column_ratio, feature_names_out=ratio_name), 
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"), 
    FunctionTransformer(np.log, feature_names_out="one-to-one"), 
    StandardScaler()
)

cluster_simil = Clustersimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False))

preprocessing = ColumnTransformer([
        ("bedrooms", ration_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ration_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ration_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)     

# -------------------------------------------------------------------
# 5. TRAIN AND TUNE THE MODEL (GRID SEARCH)
# -------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50), 
    'random_forest__max_features': randint(low=2, high=20)
}

print("4. Tuning Random Forest via RandomizedSearchCV (This will take a few minutes)...")
rnd_search = RandomizedSearchCV(
    full_pipeline, 
    param_distributions=param_distribs, 
    n_iter=10, 
    cv=3,  
    scoring='neg_root_mean_squared_error', 
    random_state=42
)

rnd_search.fit(housing, housing_labels)
final_model = rnd_search.best_estimator_

# -------------------------------------------------------------------
# 6. EVALUATE ON TEST SET
# -------------------------------------------------------------------
print("5. Evaluating final model on Test Set...")
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)

X_test = start_test_set.drop("median_house_value", axis=1)
y_test = start_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_predictions[:5])
print("\n====================================")
print("FINAL MODEL EVALUATION")
print("====================================")
print(f"Final Test RMSE: ${final_rmse:,.2f}")

# -------------------------------------------------------------------
# 7. SAVE AND VERIFY
# -------------------------------------------------------------------
joblib.dump(final_model, "my_california_housing_model.pkl")
print("\nModel saved successfully as: my_california_housing_model.pkl")
print("\nScript finished successfully!")