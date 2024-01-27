import pandas as pd
import matplotlib
import model_functions

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from data_transformation import test_transformation
from analysis_plots import confusion_plots

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
print(dtr.isna().sum() / len(dtr))

# check for unknown, non-existent etc. for all columns in order to better understand data
# fill na values in text columns with unknown
dtr = dtr.fillna("unknown")

# dtr[dtr['y'] == 'yes']['poutcome'].value_counts()/len(dtr[dtr['y'] == 'yes'])
# while poutcome has 82% nonexistent values the difference in success, when y equals yes and y equals no,
# make us keep this column and not dropping it


# remove outliers
numerical_features = ['age', 'campaign', 'duration']
for cols in numerical_features:
    Q1 = dtr[cols].quantile(0.25)
    Q3 = dtr[cols].quantile(0.75)
    IQR = Q3 - Q1

    filter_outlier = (dtr[cols] >= Q1 - 1.5 * IQR) & (dtr[cols] <= Q3 + 1.5 * IQR)
    filter_test = (X_test[cols] >= Q1 - 1.5 * IQR) & (X_test[cols] <= Q3 + 1.5 * IQR)

print(filter_outlier.value_counts() / len(filter_outlier))
print(filter_test.value_counts() / len(filter_test))
# 7% percent of data belong to above outliers in both train and test set

# remove outliers
# dtr = dtr[filter]

# transform values to float
month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
              'nov': 11, 'dec': 12}
dtr['month'] = dtr['month'].map(month_dict)

# housing default and loan preprocessing
dictionary = {'yes': 1, 'no': 0, 'unknown': -1}
binary_columns = ['housing', 'default', 'loan', 'y']
for i in binary_columns:
    dtr[i] = dtr[i].map(dictionary)

dictionary_edu = {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": -1}
dtr['education'] = dtr['education'].map(dictionary_edu)

# categorical
categorical_columns = ['job', 'marital', 'contact', 'poutcome']

dtrenc = pd.get_dummies(dtr, columns=categorical_columns, prefix=categorical_columns, drop_first=True)

dtrenc = dtrenc.replace({True: 1, False: 0})

scaling_col = ['age', 'balance', 'day_of_week', 'month', 'duration', 'campaign', 'pdays', 'previous']

dtrenc_sc = dtrenc.copy()
scaler = StandardScaler()
dtrenc_sc[scaling_col] = scaler.fit_transform(dtrenc_sc[scaling_col])

# create validation set for both scaled and unscaled data
X_train, X_val, y_train, y_val = train_test_split(dtrenc.drop(columns='y'), dtrenc['y'], test_size=0.15, random_state=0)

X_train_sc = dtrenc_sc.drop(columns='y')
y_train_sc = dtrenc_sc['y']

# get the bst logistic regression model
log_model = model_functions.logistic_train(X_train_sc, y_train_sc)

# make the same transformation for testing, except removing outliers as we have to predict for all values
X_test_sc, y_test_sc = test_transformation(X_test, y_test, True, month_dict, dictionary, binary_columns,
                                                           dictionary_edu, categorical_columns)

# predict for training and test set
y_pred_log_train, y_pred_log_test = model_functions.model_predict(log_model, X_train_sc, X_test_sc)

# accuracy_score(y_pred_log_test,y_test_sc)
print(classification_report(y_pred_log_test, y_test_sc))

log_cm = confusion_plots(y_train_sc, y_test_sc, y_pred_log_train, y_pred_log_test)

# ######XGBoost

# get the bst logistic regression model
xgb_model = model_functions.xgboost_train(X_train, y_train)

# make the same transformation for testing, except removing outliers as we have to predict for all values
X_test, y_test = test_transformation(X_test, y_test, False, month_dict, dictionary, binary_columns,
                                                     dictionary_edu, categorical_columns)

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
