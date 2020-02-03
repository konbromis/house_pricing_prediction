# -*- coding: utf-8 -*-
"""

@author: Konstantinos Bromis
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pandas import to_datetime
# Paths to csv files
path_attributes = "C:\\Users\\konstantinos\\Desktop\\Kostas\\house_price_pred\\attributes.csv"
path_geo_mapping = "C:\\Users\\konstantinos\\Desktop\\Kostas\\house_price_pred\\geoMappings.csv"
path_values = "C:\\Users\\konstantinos\\Desktop\\Kostas\\house_price_pred\\values.csv"

# Convert csv files to dataframes
attributes = pd.read_csv(path_attributes, sep= ';', engine = 'python', error_bad_lines = False)
geo_mapping= pd.read_csv(path_geo_mapping, sep= ';', engine = 'python', error_bad_lines = False)
values = pd.read_csv(path_values, sep=';', engine= 'python', error_bad_lines = False)

# Change date type
values['Date.of.Sale'] = list(map(lambda x: pd.to_datetime(x, format = '%d-%b-%y'), values['Date.of.Sale']))

# From property ids that are duplicated keep the most recent
values['Date.of.Sale'] = values.groupby('propertyID').apply(lambda x: x['Date.of.Sale'].max()).reset_index(drop = True)
values = values[:19087]

# Replace zeros from features Floor.Area and Year.Built and values >2019 & <100 in feature Year.Built
cols = ['Year.Built', 'Floor.Area']
attributes[cols] = attributes[cols].replace({0:np.nan})
attributes['Year.Built'].values[attributes['Year.Built'] > 2019] = np.nan
attributes['Year.Built'].values[attributes['Year.Built'] <= 100] = np.nan

# Merge columns from all dataframes
attributes['PCD'] = geo_mapping['PCD']
attributes['PCA'] = geo_mapping['PCA']
features = pd.merge(attributes, values, on = 'propertyID')

# Drop columns that will not be used further on
features = features.drop(['propertyID', 'Record.ID', 'Date.of.Sale'], axis = 1)
features.isna().sum()/19087

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
label_list = ['Property.Type','Energy.Rating','PCD', 'PCA']

for col in label_list: # encode data leaving nan as they are
    label_column = pd.DataFrame(features[col].values)
    temp_labels  = pd.Series([i for i in label_column.iloc[:, 0].unique() if type(i) == str])
    labelencoder_X.fit(temp_labels)
    features[col] = features[col].map(lambda x: labelencoder_X.transform([x])[0] if type(x) == str else x)

# Multiple Imputation to fill nan values   
from fancyimpute import IterativeImputer
import missingno as msno
import matplotlib.pyplot as plt

msno.bar(features, figsize=(12,6), fontsize=12, color='steelblue')
mice = IterativeImputer()
data = pd.DataFrame(data = mice.fit_transform(features), columns = features.columns, 
                        index = features.index)

# checking if there is no null value anymore
data.isnull().values.any()

# Drop rows that Year.Built>2019 & <0
data = data[~(data['Year.Built'] < 0) & ~(data['Year.Built'] > 2019)]

# Calculate age of house
data['Age_House'] = 2019 - data['Year.Built']

# Move Target variable at the end
data = data[[c for c in data if c not in ['Sale.Price']] + ['Sale.Price']]

# Drop Year.Built
data = data.drop(['Year.Built'], axis=1)

# Change types of variables
for col in label_list:
    data[col] = data[col].astype(int)
    data[col] = data[col].astype('category')

import seaborn as sns
from scipy import stats
from scipy.stats import norm
# removing outliers
data.drop(data[data['Bathrooms']>8].index, inplace=True)
data.shape

# Splitting data to training and test set
from sklearn.model_selection import train_test_split
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#separate variables into new data frames
numeric_data = X_train.select_dtypes(include=[np.number])
numeric_data = pd.concat([numeric_data, y_train], axis=1)
cat_data = X_train.select_dtypes(exclude=[np.number])

#correlation plot for numeric data
corr = numeric_data.corr()
sns.heatmap(corr)

print (corr['Sale.Price'].sort_values(ascending=False))
plt.title('Correlation between Sale Price and Bathrooms')
plt.scatter(data['Bathrooms'], data['Sale.Price'])
plt.xlabel('No of Bathrooms')
plt.ylabel('Sale Price')
plt.show()

plt.title('Correlation between Sale Price and Floor Area')
plt.scatter(data['Floor.Area'], data['Sale.Price'])
plt.xlabel('Floor Area')
plt.ylabel('Sale Price')
plt.show()

# Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
del numeric_data['Sale.Price']
numeric_features = [f for f in numeric_data.columns]
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.fit_transform(X_test[numeric_features])

# Create dummy variables for categorical variables (> 2 categories) and delete one dummy variable
onehotlist = ['Property.Type', 'Energy.Rating', 'PCD']
X_train = pd.get_dummies(X_train, prefix = onehotlist, columns = onehotlist)
X_train = X_train.drop(['Property.Type_0', 'Energy.Rating_0', 'PCD_0'], axis = 'columns')
X_test = pd.get_dummies(X_test, prefix = onehotlist, columns = onehotlist)
X_test = X_test.drop(['Property.Type_0', 'Energy.Rating_0', 'PCD_0'], axis = 'columns')

# Model Training and Evaluation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
models = [RandomForestRegressor(n_estimators=200,criterion='mse',max_depth=20,random_state=100),DecisionTreeRegressor(criterion='mse',max_depth=11,random_state=100),GradientBoostingRegressor(n_estimators=200,max_depth=12)]
learning_mods = pd.DataFrame()
temp = {}#run through models
for model in models:
    print(model)
    m = str(model)
    temp['Model'] = m[:m.index('(')]
    model.fit(X_train, y_train)
    temp['R2_Price'] = r2_score(y_test, model.predict(X_test))
    print('score on training',model.score(X_train, y_train))
    print('r2 score',r2_score(y_test, model.predict(X_test)))
    learning_mods = learning_mods.append([temp])
learning_mods.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
learning_mods.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()

