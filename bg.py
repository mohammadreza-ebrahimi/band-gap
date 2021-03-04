import numpy as np
import scipy as sc
import pandas as pd
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# %%

bg_data = pd.read_csv("C:/Users/Al-Mahdi/Dropbox/Python/band_gap_PRB93-115104.csv")
bg_data.head(10)
# %%
bg_data.info()
bg_data.describe()

# %%

bg_main = bg_data.drop(['Space-group', 'Eg(exp.;eV)', 'Type'], axis=1, inplace=False)
bg_main.head(10)

# %%
bg_main.hist(bins=50, figsize=(20, 15))
plt.show()

# %%
bg_main['PBE_cat'] = pd.cut(bg_main['Eg(PBE;eV)'],
                            bins=['-0.0001', '0.500', '1.000', '1.500', '2.000', '2.500', '3.000',
                                  '3.500', '4.000', '4.500', '5.000', '5.500', '6.000', '6.500',
                                  '7.000', '7.500', '8.000', np.inf],
                            labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                    '11', '12', '13', '14', '15', '16', '17'])
# print(bg_main['PBE_cat'])
bg_main['PBE_cat'].hist()
plt.show()

st_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in st_split.split(bg_main, bg_main['PBE_cat']):
    st_train_set = bg_main.iloc[train_index]
    st_test_set = bg_main.iloc[test_index]
# print(st_train_set)
# %%
bg_main.info()
# %%
train_set = st_train_set.copy()
test_set = st_test_set.copy()
# %%
for set_ in (test_set, train_set):
    set_.drop('PBE_cat', axis=1, inplace=True)
# %%
train_set.columns
st_train_set.columns
# %%
train_set.plot(kind='scatter', x='Eg(PBE;eV)', y='Eg(mBJ;eV)')
plt.show()

# %%
attributes = ['Eg(PBE;eV)', 'Eg(mBJ;eV)', 'Eg(G0W0;eV)']
scatter_matrix(train_set[attributes], figsize=(15, 10))
plt.show()
# %% ...
band_gap = train_set.drop('Eg(G0W0;eV)', axis=1, inplace=False)
band_gap_label = train_set['Eg(G0W0;eV)'].copy()

# %%
band_gap_num = band_gap.drop('Compound', axis=1)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std', StandardScaler()),
])
band_gap_tr = pipe.fit_transform(band_gap_num)
# %%
num_attribs = list(band_gap_num)
cat_attribs = ['Compound']

full_pipe = ColumnTransformer([
    ('num', pipe, num_attribs),
    ('cat', OrdinalEncoder(), cat_attribs)
])
band_gap_prepared = full_pipe.fit_transform(band_gap)
# %%
# OrdinalEncoder().categories
# band_gap_prepared_df = pd.DataFrame(band_gap_prepared)
# band_gap_prepared_df.head(10)
# %%
lin_reg = LinearRegression()
lin_reg.fit(band_gap_prepared, band_gap_label)
# %%
band_gap_prediction = lin_reg.predict(band_gap_prepared)
zip_sample = zip(band_gap_prediction, band_gap_label)
for i, j in zip_sample:
    print(i, j)

bg_mse = mean_squared_error(band_gap_prediction, band_gap_label)
bg_rmse = np.sqrt(bg_mse)
bg_rmse
# %%
band_gap_scores = cross_val_score(lin_reg, band_gap_prepared, band_gap_label,
                                  scoring="neg_mean_squared_error",
                                  cv=10)
band_gap_rmse_scores = np.sqrt(-band_gap_scores)


def display_score(scores):
    print("scores", scores)
    print("Mean Scores", scores.mean())
    print("Standard Deviation", scores.std())


display_score(band_gap_rmse_scores)
# %%
# for getting the Hyperparameters of specific model we can use following method:
# bg_forest = RandomForestRegressor()
# bg_forest.get_params()
# %%
# param_grid = [
#    {'n_jobs': [1, 2, 3]},
#    {'copy_X': [False], 'normalize': [True], 'n_jobs': [1, 2]}
# ]
# grid_search = GridSearchCV(lin_reg, param_grid, cv=5,
#                           scoring='neg_mean_squared_error',
#                           return_train_score=True)
# grid_search.fit(band_gap_prepared, band_gap_label)
# %%
band_gap_test = test_set.drop('Eg(G0W0;eV)', axis=1, inplace=False)
band_gap_test_labels = test_set['Eg(G0W0;eV)'].copy()
# %%

bg_test_num = band_gap_test.drop('Compound', axis=1)

test_pipe = pipe.transform(bg_test_num)
bg_test_prepared = full_pipe.fit_transform(band_gap_test)

# %%
test_prediction = lin_reg.predict(bg_test_prepared)
test_mse = mean_squared_error(test_prediction, band_gap_test_labels)
test_rmse = np.sqrt(test_mse)
test_rmse
# %%
