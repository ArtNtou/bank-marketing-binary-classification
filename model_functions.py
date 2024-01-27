import pandas as pd
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

matplotlib.use('Qt5Agg')


def xgboost_train(X_train, y_train):
    # XGBoost with validation set
    X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Hyperparameter tuning for XGBoost with validation set
    param_grid_xgb = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
    }

    grid_xgb = GridSearchCV(
        xgb.XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42),
        param_grid_xgb, cv=10, scoring='roc_auc')
    grid_xgb.fit(X_train_xgb, y_train_xgb, eval_metric='logloss', eval_set=[(X_val, y_val)], early_stopping_rounds=10,
                 verbose=False)

    # Best hyperparameters
    best_params_xgb = grid_xgb.best_params_

    # XGBoost with best hyperparameters and validation set
    best_xgb = xgb.XGBClassifier(**best_params_xgb, random_state=42)
    best_xgb.fit(X_train, y_train, eval_metric='logloss', eval_set=[(X_val, y_val)], early_stopping_rounds=10,
                 verbose=False)

    return best_xgb


def logistic_train(X_train_sc, y_train_sc):
    # Hyperparameter tuning for Logistic Regression
    param_grid_logreg = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        # 'verbose': 1
    }

    grid_logreg = GridSearchCV(LogisticRegression(random_state=42), param_grid_logreg, cv=10, scoring='roc_auc')
    grid_logreg.fit(X_train_sc, y_train_sc)

    # Logistic Regression with best hyperparameters
    best_logreg = grid_logreg.best_estimator_

    return best_logreg


# preprocessing on test data
def test_transformation(X_test, y_test, scaling, month_dict, dictionary, binary_columns, dictionary_edu,
                        categorical_columns):
    # merge them for analysis
    dtr = pd.merge(X_test, y_test, left_index=True, right_index=True)

    # checkfor uknown, non-existent etc. for all columns in order to better understand data
    # fill na values in text columns with unknown
    dtr = dtr.fillna("unknown")

    dtr['month'] = dtr['month'].map(month_dict)

    for i in binary_columns:
        dtr[i] = dtr[i].map(dictionary)

    dtr['education'] = dtr['education'].map(dictionary_edu)

    dtsenc = pd.get_dummies(dtr, columns=categorical_columns, prefix=categorical_columns, drop_first=True)

    scaling_col = ['age', 'balance', 'day_of_week', 'month', 'duration', 'campaign', 'pdays', 'previous']

    if scaling:
        dtsenc = dtsenc.copy()
        scaler = StandardScaler()
        dtsenc[scaling_col] = scaler.fit_transform(dtsenc[scaling_col])

    dtsenc = dtsenc.replace({True: 1, False: 0})

    X_test = dtsenc.drop(columns='y')
    y_test = dtsenc['y']

    return X_test, y_test


# get predictions
def model_predict(model, X_train, X_test):
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)

    return y_pred_train, y_pred_val


# get confusion matrix and plot ROC graphs and AUC values
def confusion_plots(y_train, y_test, y_pred_train, y_pred_test):
    # ROC Curve and AUC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_train)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Plot ROC Curves for tuned models
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train,
             label=f'Train (AUC = {roc_auc_train:.2f})', linestyle='--')
    plt.plot(fpr_test, tpr_test,
             label=f'Test (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Curve')
    plt.legend()
    plt.show(block=True)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show(block=True)

    return cm
