import pandas as pd
import matplotlib
from functions import model_functions

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from functions.data_transformation import train_transformation,test_transformation
from functions.analysis_plots import confusion_plots

matplotlib.use('Qt5Agg')

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Do train test split from the beginning
# Do not reach to conclusions from data that should not be available to you
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# y.value_counts()/len(y)
# train, test split. Use stratify due to class imbalance (89% no)

# merge them for analysis
dtr = pd.merge(X_train, y_train, left_index=True, right_index=True)

# check for na
# print("Percentage of NAs in each column", dtr.isna().sum() / len(dtr))

# preprocess training data
train_data_modified, transformations = train_transformation(dtr)

# columns to be scaled
scaling_col = ['age', 'balance', 'day_of_week', 'month', 'duration', 'campaign', 'pdays', 'previous']

train_data_modified_sc = train_data_modified.copy()
scaler = StandardScaler()
train_data_modified_sc[scaling_col] = scaler.fit_transform(train_data_modified_sc[scaling_col])

scaler_dict = {"scaler_model": scaler, "scaling_columns": scaling_col}

# transform dataframe to X and y
X_train, y_train = train_data_modified.drop(columns='y'), train_data_modified['y']

# Scaled data
X_train_sc = train_data_modified_sc.drop(columns='y')
y_train_sc = train_data_modified_sc['y']

# get the bst logistic regression model
log_model = model_functions.logistic_train(X_train_sc, y_train_sc)

# add scaler_dict to transformations dictionary
transformations['scaler_dict'] = scaler_dict

# make the same transformation for testing, except removing outliers as we have to predict for all values
X_test_sc, y_test_sc = test_transformation(X_test, y_test, True, transformations)

# predict for training and test set
y_pred_log_train, y_pred_log_test = model_functions.model_predict(log_model, X_train_sc, X_test_sc)

# accuracy_score(y_pred_log_test,y_test_sc)
print(classification_report(y_pred_log_test, y_test_sc))

log_cm = confusion_plots(y_train_sc, y_test_sc, y_pred_log_train, y_pred_log_test)

# ######XGBoost

# get the bst logistic regression model
xgb_model = model_functions.xgboost_train(X_train, y_train)

# make the same transformation for testing, except removing outliers as we have to predict for all values
X_test, y_test = test_transformation(X_test, y_test, False, transformations)

# predict for training and test set
y_pred_xgb_train, y_pred_xgb_test = model_functions.model_predict(xgb_model, X_train, X_test)

accuracy_score(y_pred_xgb_test, y_test)
print(classification_report(y_pred_xgb_test, y_test))

xgb_cm = confusion_plots(y_train, y_test, y_pred_xgb_train, y_pred_xgb_test)

# Get feature importance scores
feature_importance = xgb_model.feature_importances_

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
