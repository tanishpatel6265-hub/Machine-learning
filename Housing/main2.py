import sys
assert sys.version_info >= (3, 7)  #use python 3.7
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")  #
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tarfile
import urllib.request
from zlib import crc32  # convert a number into a 32bit integer number used for hashing 


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
    path = IMAGES_PATH/ f"{fig_id}.{fig_extension}"
    print(f"Saving figure {fig_id}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

plt.rc("font", size=14)
plt.rc("axes", labelsize=14 , titlesize=14)
plt.rc("legend" , fontsize=14)
plt.rc("xtick" , labelsize=10)
plt.rc("ytick" , labelsize=10)

housing.hist(bins=50, figsize=(20,8))
save_fig("attribute_histogram_plots")
# plt.show()


# -------------------------------------------SHUFFLING---------------------------------------------------

'''METHORD-1 for shuffling'''

# def shuffle_and_split_data(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]
# train_set , test_set = shuffle_and_split_data(housing , 0.2)
# print(len(train_set))
# print(len(test_set))


'''METHORD-2 for shuffling'''



# def is_id_in_test_set(identifier , test_ratio):
#      return crc32(np.int64(identifier))< test_ratio * 2**32

# def splid_data_with_id_hash(data , test_ratio , id_column):
#      ids = data[id_column]
#      in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))  #apply() runs a function on every value.
#      return data.loc[~in_test_set], data.loc[in_test_set] #.loc selects rows using labels or Boolean masks.

# housing_with_id = housing.reset_index()
# housing_with_id["id"] = housing_with_id["longitude"]*1000 + housing_with_id["latitude"]
# train_set , test_set = splid_data_with_id_hash(housing_with_id , 0.2 , "id")

'''METHORD-3 for shuffling using sklearn'''

# from sklearn.model_selection import train_test_split
# train_set , test_set = train_test_split(housing , test_size = 0.2 , random_state=42)  ## with randon

'''METHORD-4 Using sklearn'''

housing["income_cat"]= pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.,np.inf], labels = [1,2,3,4,5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot = 0, grid = True)
plt.xlabel("income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot")
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits= 10 , test_size=0.2 , random_state= 42)
start_split = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]) :
train_split_n = housing.iloc[train_index],
test_split_n = housing.iloc[test_index],
start_split.append([train_split_n,test_split_n])
"""This above function will give the 10 different train and test set """


"""If you want 1 single set so there is a shorter way """
start_train_set , start_test_set = train_test_split(housing , test_size= 0.2 ,stratify=housing["income_cat"], random_state= 42)

#----------------------------------------------data is sperated in to 2set-------------------------------------
""" THIS DISPLAY THE PROPROTION OF THE DATA"""
def income_cat_proportions(data):
return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
 "Overall %": income_cat_proportions(housing),
 "Stratified %": income_cat_proportions(start_test_set),
  "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
compare_props["Overall %"] - 1)

print((compare_props * 100).round(2))

#---------------------------------------------------------------------------------------------------------------
"""we have done some of the changes in the data like income_cat so we delete it"""

for set_ in (start_train_set , start_test_set):
set_.drop("income_cat" , axis = 1 , inplace = True)

#---------------------------------------------------------------------------------------------------------------

"""DATA VISUALIZATION """

housing = start_train_set.copy()

housing.plot(kind = "scatter" , x = "longitude" , y = "latitude" , grid = True )
save_fig("bad_visualization_plot")
# plt.show()

housing.plot(kind = "scatter" , x = "longitude" , y = "latitude" , grid = True  , alpha = 0.2 )
save_fig("better_visualization_plot")
# plt.show()

housing.plot(kind = "scatter" , x = "longitude" , y = "latitude" , grid = True , s=housing["population"]/100, label= "population" , c="median_house_value" , cmap ="jet" , colorbar = True , legend = True , sharex = False , figsize = (10,7))
save_fig("housing_prices_scatterplot")
# plt.show()

filename = "california.png"
if not (IMAGES_PATH / filename).is_file():
     homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
     url = homl3_root +  "images/end_to_end_project/" + filename
     print("Downloading" , filename)
     urllib.request.urlretrieve(url , IMAGES_PATH / filename)

housing_renamed = housing.rename(columns = {
     "latitude": "Latitude"  , "longitude" : "Longitude" , "population" : "Population" , "median_house_value" : "Median house value (USD)"
})
housing_renamed.plot(
     kind = "scatter" , x= "Longitude" , y = "Latitude" , s = housing_renamed["Population"] / 100 , label = "Population" , c="Median house value (USD)" , cmap = "jet" , colorbar = True , legend = True , sharex = False , figsize = (10,7)
)

california_img = plt.imread(IMAGES_PATH / filename)
axis = -124.55 , -113.95 , 32.45 , 42.05 
plt.imshow(california_img , extent= axis)
save_fig("california_housing_price_plot")
# plt.show()

"""DATA VISULAZATION IS OVER """
#------------------------------------------------------------------------------------------------------

corr_matrix = housing.corr(numeric_only = True)
corr_matrix["median_house_value"].sort_values(ascending = False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value" , "median_income" , "total_rooms" , "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8) )
save_fig("scatter_matrix_plot")
# plt.show()

housing.plot(kind = "scatter" , x= "median_income" ,  y = "median_house_value" , alpha= 0.1 , grid = True)
save_fig("income_vs_house_value_scatterplot")
# plt.show()

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
corr_matrix = housing.corr(numeric_only = True)
#print(corr_matrix["median_house_value"].sort_values(ascending = False))

#-----------------------------------------------------------------------------------------------------------
"""Preparing the data for ML"""
housing = start_train_set.drop("median_house_value" , axis = 1)
housing_labels = start_train_set["median_house_value"].copy()
"""this is to deal with NAN numberical data """
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include = [np.number])  # select only the numerical data
imputer.fit(housing_num)

X = imputer.transform(housing_num)
imputer.feature_names_in_            #Show the names of the columns the imputer was fitted on.

"""Now we will deal with the text data"""
# here we first use to convert the text to numberical data because machine learning data are more familear with numbers
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(8)
from sklearn.preprocessing import OrdinalEncoder  # used for converting text to interger
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:8])
print(ordinal_encoder.categories_)

# the above methord for reading text atribute is not proper because ML models learns well on the numbers

#here we are using one-Hot algorithms

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse_output=False)    # if we right sparse_output=False  we dont need to write the last command .toarray()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) 
print(housing_cat_1hot)          
print(housing_cat_1hot)  # this will show the matrix created by text atribute 

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

from sklearn.linear_model import LinearRegression
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]] , scaled_labels)
some_new_Data = housing[["median_income"]].iloc[:5]

scaled_predicitions = model.predict(some_new_Data)
predicitions = target_scaler.inverse_transform(scaled_predicitions)
print(predicitions)

from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predicitions = model.predict(some_new_Data)
print("/")

from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log , inverse_func= np.exp)
log_pop = log_transformer.transform(housing[["population"]])

print(log_pop)

rbf_transformer = FunctionTransformer(rbf_kernel , kw_args = dict(Y=[[35.]]  , gamma = 0.1 ))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

from sklearn.base import BaseEstimator , TransformerMixin
from sklearn .utils.validation import check_array , check_is_fitted

class StanderScalerClonr(BaseEstimator , TransformerMixin):
    def __init__(self , with_mean = True):
         self.with_mean = with_mean
    def fit(self , X , Y = None):
        X = check_array(X)
        self.mean_ = X.mean(axis = 0)
        self.scale_ = X.std(axis = 0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self , X):
         check_is_fitted(self)
         X = check_array(X)
         assert self.n_features_in_ == X.shape[1]
         if self.with_mean:
              X= X-self.mean_
         return X/self.scale_

from sklearn.cluster import KMeans
class Clustersimilarity(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):

        # Create the KMeans model
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )

        # Find cluster centers
        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self

    def transform(self, X):

        # Calculate similarity of each data point
        # with every cluster center
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, input_features=None):

        # Return names for the features created by transform()
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]

    
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline(["impute" , SimpleImputer(strategy= "median") , "standardize" , StandardScaler()]) # in this methord we wirte a (name , transformer)

from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy= "median") , StandardScaler()) # in this sklearn autmatically assinged name just we need to give the transformer

from sklearn import set_config
set_config(display='diagram')
num_pipeline

housing_num_prepared = num_pipeline.fit_transform(housing_num)

df_housing_num_prepared = pd.DataFrame(housing_num_prepared , columns=num_pipeline.get_feature_names_out() , index= housing_num.index)
print(df_housing_num_prepared.head(2))

from sklearn.compose import ColumnTransformer

num_attribs = ["longitude" , "latitude", "housing_median_age" , "total_rooms" , "total_bedrooms" , "population" , "households" ,"median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent") , OneHotEncoder(handle_unknown= "ignore"))
preprocessing = ColumnTransformer([("num" , num_pipeline , num_attribs) , ("cat" , cat_pipeline , cat_attribs)])

from sklearn.compose import make_column_selector,make_column_transformer
preprocessing = make_column_transformer(
(num_pipeline, make_column_selector(dtype_include=np.number)),
(cat_pipeline, make_column_selector(dtype_include=object)),
)
housing_prepared = preprocessing.fit_transform(housing)

def column_ratio(X):
     return X[:,[0]] / X[:, [1]]

def ratio_name(function_transformer , feature_name_in):
     return ["ratio"] #feature name out

def ration_pipeline():
     return make_pipeline(SimpleImputer(strategy= "median") , FunctionTransformer(column_ratio , feature_names_out=ratio_name) , StandardScaler())

log_pipeline = make_pipeline(SimpleImputer(strategy= "median") , FunctionTransformer(np.log , feature_names_out="one-to-one") ,StandardScaler())
cluster_simil = Clustersimilarity(n_clusters=10 , gamma=1. , random_state= 42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median") , StandardScaler())

preprocessing = ColumnTransformer([
     ("bedrooms" , ration_pipeline() , ["total_bedrooms" , "total_rooms"]),
     ("rooms_per_house" , ration_pipeline() , ["total_rooms" , "households"]),
     ("people_per_house" , ration_pipeline(), ["population" , "households"]),
     ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)    # one column remaining: housing_median_age     
housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)


#-------------------------------------------------------------------------------------------------------------------
"""Traning Model"""

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())

lin_reg.fit(housing, housing_labels)

housing_prediction = lin_reg.predict(housing)

print(housing_prediction[:5].round(-2))
# -2 = rounded to the nearest hundred

print(housing_labels.iloc[:5].values)
# comparing predicted values with original values


error_ratios = (
    housing_prediction[:5].round(-2)
    / housing_labels.iloc[:5].values
    - 1
)

print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))
# measures the prediction error percentage


try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)


lin_rmse = root_mean_squared_error(
    housing_labels,
    housing_prediction
)

print(lin_rmse)


from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing , DecisionTreeRegressor(random_state= 42))
tree_reg.fit(housing , housing_labels)

housing_predicition = tree_reg.predict(housing)
print(housing_predicition[:5].round(-2)) # -2 = rounded to the nearest hundredte
tree_rmse = root_mean_squared_error(housing_labels , housing_predicition)
print(tree_reg) # U will see the value come out to be 0% because decision tree model is predecting on the same data set on which it is trained
#-------------------------------------------------------------------------------------------------------------------

"""Using cross validation"""

from sklearn.model_selection import cross_val_score
tree_rmses = -cross_val_score(tree_reg , housing , housing_labels , scoring="neg_root_mean_squared_error" , cv = 10)
print(pd.Series(tree_rmses).describe())

lin_rmses = -cross_val_score(lin_reg , housing , housing_labels , scoring = "neg_root_mean_squared_error" , cv = 10)
print(pd.Series(lin_rmses).describe())

from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing , RandomForestRegressor(random_state= 42))
forest_rmses = -cross_val_score(forest_reg , housing , housing_labels , scoring="neg_root_mean_squared_error" , cv = 10)
print(pd.Series(forest_rmses).describe())
forest_reg.fit(housing , housing_labels)
housing_predicitions = forest_reg.predict(housing)
forest_rmse = root_mean_squared_error(housing_labels , housing_predicitions)
print(forest_rmse)

from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
     ("preprocessing" , preprocessing),
     ("random_forest" , RandomForestRegressor(random_state= 42)),
])
param_grid = [
     {'preprocessing__geo__n_clusters': [5 , 8, 10] , 
      'random_forest__max_features' : [4 , 6,8]},
     {'preprocessing__geo__n_clusters': [10 , 15] , 
      'random_forest__max_features' : [6,8,10]}
]
grid_search = GridSearchCV(full_pipeline , param_grid ,cv = 3 , scoring='neg_root_mean_squared_error')
grid_search.fit(housing , housing_labels)
print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score" , ascending=False , inplace = True)


cv_res = cv_res[["param_preprocessing__geo__n_clusters", "param_random_forest__max_features" , "split0_test_score", "split1_test_score" ,  "split2_test_score" ,"mean_test_score" ]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

print(cv_res.head())

"""Randomized Search"""

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'preprocessing__geo__n_clusters' : randint(low=3, high = 50), 
                  'random_forest__max_features': randint(low = 2 , high = 20)}

rnd_search = RandomizedSearchCV(
     full_pipeline, param_distributions=param_distribs , n_iter=10 , cv=3,  scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing , housing_labels)

cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
print(cv_res.head())

# ============================================================
# FINAL MODEL EVALUATION ON THE TEST SET
# ============================================================

final_model = rnd_search.best_estimator_

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importances = final_model["random_forest"].feature_importances_

print("\nFeature Importances:")
print(feature_importances.round(2))

sorted_feature_importances = sorted(
    zip(
        feature_importances,
        final_model["preprocessing"].get_feature_names_out()
    ),
    reverse=True
)

print("\nSorted Feature Importances:")
for importance, feature_name in sorted_feature_importances:
    print(f"{feature_name}: {importance:.4f}")


# ------------------------------------------------------------
# FINAL TEST SET EVALUATION
# ------------------------------------------------------------

X_test = start_test_set.drop("median_house_value", axis=1)

y_test = start_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(
    y_test,
    final_predictions
)

print("\n====================================")
print("FINAL MODEL EVALUATION")
print("====================================")
print(f"Final Test RMSE: {final_rmse:.2f}")


# ------------------------------------------------------------
# SHOW SOME ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

print("\nFirst 5 Predictions:")

results = pd.DataFrame({
    "Actual Price": y_test.iloc[:5].values,
    "Predicted Price": final_predictions[:5]
})

print(results.round(2))


# ------------------------------------------------------------
# SAVE THE MODEL
# ------------------------------------------------------------

import joblib

joblib.dump(
    final_model,
    "my_california_housing_model.pkl"
)

print("\nModel saved successfully as:")
print("my_california_housing_model.pkl")


# ------------------------------------------------------------
# LOAD THE SAVED MODEL
# ------------------------------------------------------------

final_model_reloaded = joblib.load(
    "my_california_housing_model.pkl"
)

# Pretend these are new districts
new_data = X_test.iloc[:5]

predictions = final_model_reloaded.predict(new_data)

print("\n====================================")
print("PREDICTIONS FROM RELOADED MODEL")
print("====================================")

print(predictions.round(2))


So first of all remove the errors and i have to make this mode more accurate so what i have to do dont give direct code suggest me 